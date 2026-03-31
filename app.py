import streamlit as st
import google.generativeai as genai
import os
import re

# --- API CONFIGURATION ---
# Supports multiple comma-separated keys for rotation (triples effective daily quota)
_raw_keys = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not API_KEYS:
    st.error("GEMINI_API_KEY not found. Set it as an environment variable or in .streamlit/secrets.toml")
    st.stop()
genai.configure(api_key=API_KEYS[0])

# --- SYSTEM INSTRUCTION (XML-structured for token efficiency) ---
SYSTEM_INSTRUCTION = """
<objective>
Generate semester-end conduct remarks for Singapore primary school students.
Write as an experienced, caring form teacher — warm, personal, and encouraging, never robotic or formulaic.
Tone: positive, growth-mindset. Lead with strengths; frame improvements gently ("can do even better if…").
Each remark: 55–75 words. All sentences describe the student in third person EXCEPT the final sentence, which is direct encouragement addressed to the student. Vary sentence structures widely across remarks.
</objective>

<input_format>
Input begins with pronouns (she/her or he/him) applying to all subsequent students until changed.
Per student: index_number descriptors (optional_parenthetical_info) score_1-5.
- Index number (e.g. 01, 02): use as student identifier. Start each remark with the index.
- Descriptors: space-separated traits guiding remark content.
- Parenthetical info: roles, awards, or improvement areas. Incorporate in 2nd or 3rd sentence, never 1st.
- Score 1–5: conduct grade (5=Excellent). Higher = more effusive praise.
- Default mode: BLGPS. Mode is specified per message as CURRENT MODE.
</input_format>

<framework_rules>
Map input descriptors to the active framework. Do not take descriptors literally — find best-fit framework terms and use them as a word bank.

<blgps>
USE ONLY WHEN MODE = BLGPS. Never use category titles directly — reference their traits instead.

Curious Thinker: Displays lively curiosity and desire to learn through insightful questions. Exercises critical thinking with sound reasoning, decision-making, and metacognition. Shows adaptive thinking by assessing contexts and adjusting perspectives. Demonstrates innovative thinking by generating novel ideas and refining them into solutions.
Descriptors: inquisitive, diligent, bright, meticulous, sharp, quick-witted, curious, asks questions, participates actively, hardworking.

Confident Learner: Shows motivation to learn, self-directedness, resilience to overcome challenges, and connectedness to learn well with others through effective communication and collaboration.
Descriptors: hardworking, personable, outgoing, responsible, good leader, kind, helpful, confident, self-directed, self-motivated.

Compassionate Contributor: Works cooperatively with harmony, adopts care and integrity in daily interactions, demonstrates respect in communication and action, contributes to community through responsibility and resilience, shows civic/global/cross-cultural awareness.
Descriptors: team-player, cheerful, personable, well-liked, kind, caring.

If input descriptors relate to any category, reference that category’s traits liberally.
</blgps>

<21cc>
USE ONLY WHEN MODE = 21CC. Draw from all three lists:

21st Century Competencies (include at least one if input aligns):
Critical thinking, Adaptive thinking, Inventive thinking, Communication skills, Collaboration skills, Information skills, Civic literacy, Global literacy, Cross-cultural literacy.

R3ICH Values: Respect, Responsibility, Resilience, Integrity, Care, Harmony.

Social-Emotional Competencies: Self-awareness, Self-management, Responsible decision making, Social awareness, Relationship management.
</21cc>

Both modes share 21CC and R3ICH lists as supplementary word banks.
</framework_rules>

<formatting_constraints>
- 55–75 words per remark. Vary sentence structure actively.
- All sentences third person EXCEPT final sentence: direct encouragement to student.
- Positive, upbeat tone. Growth-mindset framing for improvements.
- Parenthetical info in 2nd or 3rd sentence only.
- Every sentence must be grounded in the student's specific descriptors. AVOID generic filler that could apply to anyone, such as: "valued member of the class", "admirable qualities", "dynamic learning environment", "contributions are always valued", "remarkable sense of awareness".
- Reference framework traits naturally within your sentences. Do NOT use clunky phrasing like "embodies the spirit of", "demonstrates the qualities of", or "possessing the traits of". Weave the traits into observations about the student.
- Generate fresh language for each student. Do NOT reuse phrasing from the provided examples.
- You MUST wrap any extrapolated actions or specific examples not explicitly found in the input descriptors in single asterisks (*like this*). Do NOT italicize direct synonyms or close paraphrases. Example: input "hardworking" → "diligent" is a synonym (no asterisks), but "meticulous in completing assignments" is an extrapolation (*meticulous in completing assignments*).

Required Footer:
- BLGPS mode: **Note that the output from here is a first draft, and will always require editing as the AI is not capable of producing flawlessly accurate output. Pay special attention to whether the descriptors match the BLGPS student outcome mentioned in the remark, if any. Italicized phrases are interpretive and should be reviewed for accuracy.**
- 21CC mode: **Note that the output from here is a first draft, and will always require editing as the AI is not capable of producing flawlessly accurate output. Italicized phrases are interpretive and should be reviewed for accuracy.**
</formatting_constraints>
"""

# --- FEW-SHOT EXEMPLARS (fed as conversation history) ---
EXEMPLARS = [
    # she/her — Compassionate Contributor, score 4, simple
    ("she/her 01 softspoken compassionate 4", "01 is a soft-spoken and kind-hearted student. Possessing the traits of a compassionate contributor, she consistently shows care and empathy in her interactions with her peers. Her *gentle disposition* and *sincerity* are admirable qualities that create a positive and harmonious classroom environment. She is a valued member of the class. Keep up the wonderful attitude, 01!"),
    # she/her — Confident Learner, score 4, parenthetical improvement
    ("she/her 02 cheerful participative (can be more consistent in attendance) 4", "A cheerful and participative student, 02 brings *a great deal of positive energy* to the classroom. She *engages readily in discussions and group activities*, demonstrating the qualities of a confident learner. Her contributions are always valued by her peers and teachers alike. She is encouraged to maintain consistent attendance to maximise her learning opportunities. You are on the right track, 02!"),
    # she/her — Compassionate Contributor, score 5, many descriptors
    ("she/her 03 responsible compassionate considerate observant 5", "03 is a responsible and compassionate student who demonstrates a remarkable sense of awareness. As a compassionate contributor, she is considerate of the needs of others and is always observant, *often being the first to offer assistance*. Her *integrity* and strong sense of responsibility make her a trusted and *respected role model* in class. We are very proud of you, 03!"),
    # she/her — Curious Thinker, score 5 — DEMONSTRATES ASTERISK FORMATTING
    ("she/her 05 diligent bright driven 5", "05 is a diligent and bright student with a remarkable drive to succeed. A curious thinker, she approaches her work with *meticulous care* and is consistently self-motivated to produce work of the highest quality. Her *inquisitive nature* and inventive thinking allow her to grasp complex concepts with ease and to excel in her studies. Well done, 05!"),
    # she/her — Confident Learner, score 4, parenthetical improvement
    ("she/her 16 dependable reserved (speak up more) 4", "16 is a dependable student with a *thoughtful* and reserved nature. *Her work is always completed with care*, showing a strong sense of *responsibility*. To further develop her communication skills as a confident learner, she is encouraged to share her *insightful ideas* more frequently during class discussions. Her perspective is valuable. You can do it, 16!"),
    # she/her — Confident Learner, score 5, model student
    ("she/her 18 dependable model student driven sincere 5", "18 truly exemplifies what it means to be a model student. She is a driven and sincere individual who is exceptionally dependable. As a confident learner, her *integrity* and *commitment to excellence* are evident in all she does, *inspiring her peers to also strive for their best*. She consistently demonstrates *responsibility* and a strong work ethic. We are proud of you, 18!"),
    # he/him — Confident Learner, score 4, simple
    ("he/him 20 reliable dependable hardworking 4", "20 is a reliable and dependable student who approaches his studies with a commendable work ethic. He *takes responsibility for his learning* and can always be trusted to *complete his tasks diligently*. His *consistency and commitment* are the hallmarks of a confident learner, and these traits will serve him well as he continues to progress. Keep up the great work, 20!"),
    # he/him — Compassionate Contributor, score 5, many descriptors
    ("he/him 23 trustworthy reliable dependable helpful sincere 5", "23 is an exceptionally trustworthy and reliable student. His *integrity is beyond reproach*, and he is a sincere and helpful member of the class. As a compassionate contributor, he can always be depended upon to *assist his peers and teachers*, demonstrating a strong sense of *responsibility* and care for *his community*. It is a joy to have him in class. Well done, 23!"),
    # he/him — Curious Thinker, score 3, parenthetical improvement
    ("he/him 25 bright outspoken (can work harder) 3", "25 is a bright and outspoken student who is *confident in sharing his ideas*. His willingness to speak up shows great potential for *communication and leadership*. He can unlock his full capabilities as a curious thinker by applying himself more consistently to his studies and *channelling his intelligence into diligent work*. Continue to work hard, 25!"),
    # he/him — Confident Learner + Compassionate Contributor, score 5, leadership
    ("he/him 34 reliable good leader compassionate 5", "34 is a reliable and compassionate student who has demonstrated *excellent leadership qualities*. A confident learner, he takes on responsibilities with *maturity* and guides his peers with care and *respect*. His ability to *lead with empathy* makes him an effective and trusted leader, embodying the spirit of a compassionate contributor. We are very proud of you, 34!"),
    # he/him — score 3, parenthetical improvement, needs collaboration
    ("he/him 37 outspoken distracted (could work better with others) 3", "37 is an outspoken student with *a great deal of energy*. His willingness to voice his opinion is a strength. To grow as a learner, he would benefit from *channelling his energy into focused tasks* and developing his collaboration skills. By learning to work more *harmoniously and effectively* with his peers, he will make more positive contributions. Keep trying, 37!"),
    # she/her — Compassionate Contributor, score 4, participative
    ("she/her 40 easygoing friendly participative 4", "40 is an easygoing and friendly student who participates actively in class. Her *approachable nature* makes her a wonderful team member, and she embodies the spirit of a compassionate contributor. She *communicates well with her peers* and her willingness to contribute to discussions helps to create a *dynamic and collaborative learning environment*. Keep up the good work, 40!"),
]

# --- APP LOGIC ---
def _build_history(mode):
    """Build Gemini chat history from exemplar pairs."""
    history = []
    for user_input, model_output in EXEMPLARS:
        history.append({"role": "user", "parts": [f"CURRENT MODE: {mode}\n\nUSER INPUT: {user_input}"]})
        history.append({"role": "model", "parts": [model_output]})
    return history

def call_gemini(user_text, current_mode):
    errors = []
    for attempt in range(len(API_KEYS)):
        key_idx = (st.session_state.get("_key_idx", 0) + attempt) % len(API_KEYS)
        genai.configure(api_key=API_KEYS[key_idx])
        try:
            model = genai.GenerativeModel(
                model_name="gemini-3.1-flash-lite-preview",
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config=genai.GenerationConfig(temperature=0.55)
            )
            chat = model.start_chat(history=_build_history(current_mode))
            formatted_input = f"CURRENT MODE: {current_mode}\n\nUSER INPUT: {user_text}"
            response = chat.send_message(formatted_input)
            text = response.text
            text = re.sub(r'(?<!\n)\n(\*{0,2}Note that)', r'\n\n\1', text)
            # Advance round-robin for next call
            st.session_state["_key_idx"] = (key_idx + 1) % len(API_KEYS)
            return text
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                errors.append(str(e))
                continue
            return f"⚠️ API Error: {e}"
    return f"⚠️ Rate limited on all API keys. Last error: {errors[-1][:200] if errors else 'unknown'}"

def _assemble_structured_input(students, pronouns):
    """Convert structured form data into API-safe text with index placeholders.
    Returns (api_text, name_map) where name_map = {"S01": "Real Name", ...}."""
    name_map = {}
    parts = []
    for i, s in enumerate(students, start=1):
        idx = f"S{i:02d}"
        name_map[idx] = s["name"]
        tokens = [idx]
        if s.get("characteristics"):
            tokens.append(s["characteristics"])
        if s.get("roles"):
            tokens.append(f"({s['roles']})")
        if s.get("awards"):
            tokens.append(f"({s['awards']})")
        if s.get("other"):
            tokens.append(f"({s['other']})")
        tokens.append(str(s["rating"]))
        parts.append(" ".join(tokens))
    api_text = f"{pronouns} " + ", ".join(parts)
    return api_text, name_map

def _restore_names(text, name_map):
    """Replace index placeholders (S01, S02, ...) with real student names."""
    for idx, name in name_map.items():
        text = re.sub(r'\b' + re.escape(idx) + r'\b', name, text)
    return text

def _remarks_to_txt(text):
    """Return remarks as plain text for download, excluding the footer note."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    cleaned = []
    for para in paragraphs:
        if para.startswith("**Note") or para.startswith("Note that"):
            continue
        cleaned.append(para)
    return "\n\n".join(cleaned)

def _extract_indices(text):
    """Extract 2+ digit student index numbers from Quick Entry input."""
    return sorted(set(re.findall(r'\b(\d{2,})\b', text)), key=lambda x: int(x))

# --- SAMPLE DATA ---
_SAMPLE_QUICK = ("she/her 01 softspoken compassionate 4, 02 cheerful participative "
    "(can be more consistent in attendance) 4, 03 responsible compassionate "
    "considerate observant 5, 04 easygoing helpful participative 4, "
    "05 diligent bright driven 5, 06 responsible dependable 4, "
    "07 holds herself to a high standard considerate model student 4, "
    "08 hardworking sincere reliable 4, 09 hardworking dependable (quiet) 4, "
    "10 respectful participative hardworking 4, 11 cheerful friendly confident 4, "
    "12 softspoken hardworking 4, 13 resilient well-liked compassionate 4, "
    "14 softspoken sincere kind 4, 15 reliable hardworking 4, "
    "16 dependable reserved (speak up more) 4, 17 cheerful resilient confident 4, "
    "18 dependable model student driven sincere 5, "
    "19 compassionate caring hardworking 5, "
    "he/him 20 reliable dependable hardworking 4, "
    "22 friendly cheerful (can focus better) 4, "
    "23 trustworthy reliable dependable helpful sincere 5, "
    "24 easygoing friendly (could work better with others) 3, "
    "25 bright outspoken (can work harder) 3, "
    "27 hardworking considerate (could focus better) 4, "
    "28 reliable easygoing helpful 5, 29 hardworking driven sincere 5, "
    "30 respectful disciplined (could speak up more) 4, "
    "31 helpful sincere (could work better with others) 4, "
    "32 outspoken cheerful (could focus better) 3, "
    "33 insightful inquisitive (could work better with others) 3, "
    "34 reliable good leader compassionate 5, "
    "35 respectful dependable (could work on confidence) 3, "
    "36 hardworking softspoken reliable 5, "
    "37 outspoken distracted (could work better with others) 3, "
    "38 respectful hardworking dependable 4, "
    "39 cheerful playful outspoken (could focus better) 3 "
    "she/her 40 easygoing friendly participative 4")

_SAMPLE_NAMES = [
    {"name": "Aisha", "chars": "softspoken compassionate", "other": "", "rating": 4},
    {"name": "Mei Ling", "chars": "cheerful participative", "other": "can be more consistent in attendance", "rating": 4},
    {"name": "Priya", "chars": "responsible compassionate considerate observant", "other": "", "rating": 5},
    {"name": "Sarah", "chars": "easygoing helpful participative", "other": "", "rating": 4},
    {"name": "Li Wen", "chars": "diligent bright driven", "other": "", "rating": 5},
    {"name": "Nurul", "chars": "responsible dependable", "other": "", "rating": 4},
    {"name": "Hui Min", "chars": "holds herself to a high standard considerate model student", "other": "", "rating": 4},
    {"name": "Kavya", "chars": "hardworking sincere reliable", "other": "", "rating": 4},
    {"name": "Xin Yi", "chars": "hardworking dependable", "other": "quiet", "rating": 4},
    {"name": "Farah", "chars": "respectful participative hardworking", "other": "", "rating": 4},
    {"name": "Jia Xuan", "chars": "cheerful friendly confident", "other": "", "rating": 4},
    {"name": "Siti", "chars": "softspoken hardworking", "other": "", "rating": 4},
    {"name": "Annabel", "chars": "resilient well-liked compassionate", "other": "", "rating": 4},
    {"name": "Rui En", "chars": "softspoken sincere kind", "other": "", "rating": 4},
    {"name": "Zhi Ting", "chars": "reliable hardworking", "other": "", "rating": 4},
    {"name": "Aisyah", "chars": "dependable reserved", "other": "speak up more", "rating": 4},
    {"name": "Chloe", "chars": "cheerful resilient confident", "other": "", "rating": 4},
    {"name": "Wei Xuan", "chars": "dependable model student driven sincere", "other": "", "rating": 5},
    {"name": "Amira", "chars": "compassionate caring hardworking", "other": "", "rating": 5},
    {"name": "Rachel", "chars": "easygoing friendly participative", "other": "", "rating": 4},
]

def _load_sample_quick():
    st.session_state.quick_input = _SAMPLE_QUICK

def _load_sample_names():
    st.session_state.pronouns_radio = "she/her"
    st.session_state.num_students_input = len(_SAMPLE_NAMES)
    for i, s in enumerate(_SAMPLE_NAMES):
        st.session_state[f"name_{i}"] = s["name"]
        st.session_state[f"chars_{i}"] = s["chars"]
        st.session_state[f"roles_{i}"] = ""
        st.session_state[f"awards_{i}"] = ""
        st.session_state[f"other_{i}"] = s.get("other", "")
        st.session_state[f"rating_{i}"] = s["rating"]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Teacher Remark Assistant", layout="wide")

# --- CUSTOM THEME ---
st.markdown("""
<style>
    /* Dark navy background */
    .stApp { background-color: #020617; color: #e2e8f0; font-family: 'Century Gothic', sans-serif; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    section[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    
    /* Headers */
    h1 { color: #f59e0b !important; }
    h2, h3 { color: #2dd4bf !important; }
    .stMarkdown h1 { color: #f59e0b !important; }
    .stMarkdown h2, .stMarkdown h3 { color: #2dd4bf !important; }
    
    /* Text areas and inputs */
    .stTextArea textarea {
        background-color: #1e293b !important; color: #e2e8f0 !important;
        border: 1px solid #334155 !important; border-radius: 8px !important;
    }
    .stTextArea textarea:focus { border-color: #2dd4bf !important; box-shadow: 0 0 0 1px #2dd4bf !important; }
    
    /* Primary button */
    .stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #2dd4bf, #14b8a6) !important;
        box-shadow: 0 4px 15px rgba(45, 212, 191, 0.3) !important;
    }
    
    /* Secondary buttons */
    .stButton > button {
        background-color: #1e293b !important; color: #2dd4bf !important;
        border: 1px solid #334155 !important; border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background-color: #334155 !important; border-color: #2dd4bf !important;
    }
    
    /* Radio buttons */
    .stRadio label { color: #e2e8f0 !important; }
    .stRadio div[role="radiogroup"] label span { color: #cbd5e1 !important; }
    
    /* Alerts and info boxes */
    .stAlert { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
    div[data-testid="stNotification"] { background-color: #1e293b !important; }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #1e293b !important; color: #fbbf24 !important;
        border: 1px solid #fbbf24 !important; border-radius: 8px !important;
    }
    .stDownloadButton > button:hover { background-color: #fbbf24 !important; color: #020617 !important; }
    
    /* Spinner */
    .stSpinner > div { border-top-color: #2dd4bf !important; }
    
    /* Divider */
    hr { border-color: #1e293b !important; }
    
    /* Caption */
    .stCaption, small { color: #64748b !important; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #020617; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }
    
    /* Smooth message fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .element-container { animation: fadeIn 0.3s ease-out forwards; }

    /* Text inputs, selectbox, number input — dark theme */
    .stTextInput input, .stNumberInput input {
        background-color: #1e293b !important; color: #e2e8f0 !important;
        border: 1px solid #334155 !important; border-radius: 8px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2dd4bf !important; box-shadow: 0 0 0 1px #2dd4bf !important;
    }
    .stSelectbox > div > div { background-color: #1e293b !important; color: #e2e8f0 !important; }
    .stSelectbox [data-baseweb="select"] > div { background-color: #1e293b !important; border-color: #334155 !important; }

    /* Sidebar button hover — pink/magenta accent */
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #831843 !important;
        border-color: #ec4899 !important;
        color: #f9a8d4 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #0f172a; border-radius: 8px; gap: 2px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; background-color: transparent; }
    .stTabs [aria-selected="true"] { color: #2dd4bf !important; background-color: #1e293b !important; border-radius: 6px; }

    /* Expander */
    .streamlit-expanderHeader { background-color: #1e293b !important; color: #e2e8f0 !important; border-radius: 8px !important; }
    details { border: 1px solid #334155 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# Session state to persist generated remarks across reruns
if "last_remarks" not in st.session_state:
    st.session_state.last_remarks = ""
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "name_map" not in st.session_state:
    st.session_state.name_map = {}
if "quick_indices" not in st.session_state:
    st.session_state.quick_indices = []
if "_key_idx" not in st.session_state:
    st.session_state["_key_idx"] = 0

st.markdown("<h1 style='color: #f59e0b !important;'>🇸🇬 Student Remarks Assistant</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode_selection = st.radio("Target Mode:", ["BLGPS Mode", "21CC Mode"],
                               help="BLGPS Mode uses Boon Lay Garden Primary School's Curious Thinker / Confident Learner / Compassionate Contributor framework. 21CC Mode uses MOE Singapore's 21st Century Competencies, R3ICH values and social-emotional competencies.")
    st.divider()
    st.subheader("Special Actions")
    help_clicked = st.button("📖 Guide / Instructions",
                             help="Shows a full usage guide: input format, modes, and features.")
    st.caption("ℹ️ Italicized phrases in generated remarks are interpretive — review them for accuracy.")

# --- INPUT AREA (Tabs) ---
tab_quick, tab_names = st.tabs(["⚡ Quick Entry", "📝 Names Enabled"])

with tab_quick:
    st.button("📋 Load Sample Data", on_click=_load_sample_quick, key="sample_quick",
              help="Populates the text box with a full 38-student sample dataset for testing.")
    user_data_input = st.text_area("Input Student Details:",
                                   placeholder="she/her 01 responsible (Class Monitor) 5, 02 helpful 4",
                                   height=150, key="quick_input",
                                   help="Format: start with pronouns (she/her or he/him), then for each student: index number, descriptors, optional (parenthetical info), and a conduct score 1-5. Separate students with commas.")
    if st.button("🚀 Generate Remarks", type="primary", key="gen_quick",
                 help="Sends the input to Gemini AI and generates one remark per student."):
        if user_data_input:
            with st.spinner("Generating..."):
                res = call_gemini(user_data_input, mode_selection)
                st.session_state.last_remarks = res
                st.session_state.last_input = user_data_input
                st.session_state.name_map = {}
                st.session_state.quick_indices = _extract_indices(user_data_input)
            st.toast("✅ Remarks generated!")
        else:
            st.error("Please enter student data.")

with tab_names:
    st.caption("🔒 Student names are kept private — only index placeholders are sent to the AI.")
    st.button("📋 Load Sample Data", on_click=_load_sample_names, key="sample_names",
              help="Fills in 20 sample she/her students with placeholder names and descriptors for testing. Use this to see how the Names Enabled workflow works before entering real data.")
    pronouns = st.radio("Pronouns for this batch:", ["she/her", "he/him"], horizontal=True, key="pronouns_radio",
                        help="All students in this batch share the same pronouns. To mix pronouns, generate one batch with she/her and another with he/him, then combine the results.")
    num_students = st.number_input("Number of students:", min_value=1, max_value=45, value=1, step=1, key="num_students_input",
                                   help="How many students in this batch? Each gets their own entry form below. For a typical class of 40, you may want to split into two batches by pronoun.")
    st.caption("Rating guide: 5 = Excellent · 4 = Very Good · 3 = Good · 2 = Satisfactory · 1 = Needs Improvement")

    students = []
    for i in range(int(num_students)):
        with st.expander(f"Student {i + 1}", expanded=(i == 0)):
            name = st.text_input("Name", key=f"name_{i}", placeholder="e.g. Wei Lin",
                               help="Student's real name — kept 100% private, never sent to the AI. The name is replaced with an index placeholder (e.g. S01) before the API call, then swapped back into the final output on your device.")
            characteristics = st.text_input("Characteristics / Descriptors", key=f"chars_{i}",
                                            placeholder="e.g. cheerful responsible hardworking",
                                            help="Space-separated traits that describe this student's conduct and character. Examples: softspoken, compassionate, diligent, bright, resilient, well-liked, participative. These are the primary drivers of the remark content. Use 2–5 descriptors for best results.")
            roles = st.text_input("Class / Leadership Roles (optional)", key=f"roles_{i}",
                                  placeholder="e.g. Class Monitor, Prefect",
                                  help="Leadership positions or class duties. These are mentioned in the remark as evidence of responsibility or confidence. Leave blank if none.")
            awards = st.text_input("Awards (optional)", key=f"awards_{i}",
                                   placeholder="e.g. Good Progress Award",
                                   help="Awards or recognitions received this semester. Woven into the remark as an achievement. Leave blank if none.")
            other = st.text_input("Other Information (optional)", key=f"other_{i}",
                                  placeholder="e.g. frequent latecoming, can focus better",
                                  help="Areas for improvement or extra context. The AI phrases these gently and positively (e.g. 'can focus better' becomes 'is encouraged to improve focus'). Leave blank if purely positive.")
            rating = st.selectbox("Behaviour Rating", options=[5, 4, 3, 2, 1], key=f"rating_{i}",
                                  help="Controls how effusive the remark is. 5 = Excellent (most praise, ~75 words), 4 = Very Good, 3 = Good (balanced, includes growth areas), 2 = Satisfactory, 1 = Needs Improvement.")
            students.append({
                "name": name.strip(),
                "characteristics": characteristics.strip(),
                "roles": roles.strip(),
                "awards": awards.strip(),
                "other": other.strip(),
                "rating": rating,
            })

    if st.button("🚀 Generate Remarks", type="primary", key="gen_names",
                 help="Sends anonymised data to Gemini AI. Student names are replaced with placeholders and restored in the final output."):
        missing = [i + 1 for i, s in enumerate(students) if not s["name"]]
        if missing:
            st.error(f"Please enter a name for student(s): {', '.join(str(m) for m in missing)}")
        else:
            with st.spinner("Generating..."):
                api_text, name_map = _assemble_structured_input(students, pronouns)
                res = call_gemini(api_text, mode_selection)
                named_res = _restore_names(res, name_map)
                st.session_state.last_remarks = named_res
                st.session_state.last_input = api_text
                st.session_state.name_map = name_map
                st.session_state.quick_indices = []
            st.toast("✅ Remarks generated!")

# --- PERSISTENT OUTPUT ---
if st.session_state.last_remarks:
    st.divider()
    st.markdown("### Generated Remarks")
    st.write(st.session_state.last_remarks)

    # Per-remark copy buttons
    remarks_text = st.session_state.last_remarks
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', remarks_text) if p.strip()]
    remark_paragraphs = [p for p in paragraphs if not (p.startswith("**Note") or p.startswith("Note that"))]
    if len(remark_paragraphs) > 1:
        with st.expander("📋 Copy individual remarks"):
            for i, remark in enumerate(remark_paragraphs):
                label = remark[:40].rstrip() + "..." if len(remark) > 40 else remark
                st.code(remark, language=None)

    txt_data = _remarks_to_txt(st.session_state.last_remarks)
    col_dl, col_count = st.columns([1, 3])
    with col_dl:
        st.download_button("📥 Download TXT", txt_data, file_name="student_remarks.txt", mime="text/plain",
                           help="Downloads the remarks as a plain text file (one paragraph per student, footer excluded).")
    with col_count:
        st.caption(f"{len(remark_paragraphs)} remark(s) generated")
    with st.expander("📋 Copy all raw text"):
        st.code(st.session_state.last_remarks, language=None)

    # --- NAME SUBSTITUTION (Quick Entry) ---
    if st.session_state.quick_indices:
        with st.expander("✏️ Input Student Names", expanded=False):
            st.caption("Map index numbers to real names — this is local, nothing is sent to AI.")
            indices = st.session_state.quick_indices
            cols_per_row = 4
            for row_start in range(0, len(indices), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    pos = row_start + j
                    if pos < len(indices):
                        with row_cols[j]:
                            st.text_input(f"#{indices[pos]}", key=f"qname_{indices[pos]}",
                                          placeholder="Name")
            if st.button("✅ Apply Names", type="primary", key="apply_names",
                         help="Replaces index numbers with the names you entered. Pronouns are kept as-is."):
                name_map = {}
                for idx in indices:
                    n = st.session_state.get(f"qname_{idx}", "").strip()
                    if n:
                        name_map[idx] = n
                if name_map:
                    st.session_state.last_remarks = _restore_names(
                        st.session_state.last_remarks, name_map)
                    st.session_state.name_map = name_map
                    st.session_state.quick_indices = []
                    st.rerun()
                else:
                    st.warning("Enter at least one name to apply.")

# Process Sidebar Actions
if help_clicked:
    st.divider()
    st.markdown("### 📖 Usage Guide")
    st.info("""
**Input Format (Quick Entry)**

Start with pronouns (`she/her` or `he/him`), then for each student provide:
- **Index number** (e.g. 01, 02)
- **Descriptors** — space-separated traits (e.g. `responsible hardworking cheerful`)
- **Parenthetical info** *(optional)* — roles, awards, or areas for improvement (e.g. `(Class Monitor)`, `(can focus better)`)
- **Conduct score** — 1 to 5 (5 = Excellent)

Separate students with commas. Switch pronouns mid-input by writing the new pronoun before the next student.

**Example:**
`she/her 01 responsible hardworking (Class Monitor) 5, 02 cheerful friendly 4, he/him 03 bright outspoken (can work harder) 3`

**Modes**
- **BLGPS Mode** — maps to Curious Thinker / Confident Learner / Compassionate Contributor framework
- **21CC Mode** — maps to MOE Singapore's 21st Century Competencies, R3ICH values, and social-emotional competencies

**Names Enabled Tab**
Use the structured form to enter student names privately — names never leave your browser and are swapped into the output locally.

**Italicized Phrases**
Phrases in *italics* in the output are interpretive extrapolations beyond the input descriptors. Review these for accuracy before finalising.

**After Generating**
- Download remarks as a TXT file
- Copy individual remarks or the full raw text
- In Quick Entry, use "Input Student Names" to map index numbers to real names locally
""")

st.divider()
st.caption("Powered by Gemini 3.1 Flash Lite Preview | Framework: Singapore MOE 21CC / BLGPS")


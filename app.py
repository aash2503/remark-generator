import streamlit as st
import google.generativeai as genai
import os
import re

# --- API CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("GEMINI_API_KEY not found. Set it as an environment variable or in .streamlit/secrets.toml")
    st.stop()
genai.configure(api_key=API_KEY)

# --- THE COMPLETE SYSTEM INSTRUCTION ---
SYSTEM_INSTRUCTION = """
At the end of each semester at our school in Singapore, students receive a results slip which contains a summary of their academic progress, but also a short set of comments/remarks which briefly details their general conduct, dispositions and aptitudes. It may even include a gently and kindly phrased comment that identifies a potential area for improvement for the student. You will serve as my assistant for this task, the generation of these comments. These comments include certain general notes, and are provided to the students once every six months. As such, the overall tone, thematic notes and structure should be broadly cross-compatible, but successive sets of remarks for the same student should not be too similar to the point of being repetitive when they are read consecutively. Note that, in all cases, start with positive elements but then bring in areas for improvement as a secondary point. 

I will provide samples of how the comments should look. I will then provide guidelines for generating these comments, which should apply across the board in every case where comments are to be generated. 
Before I provide the samples, note that you are to focus on the formatting. In particular, consider the variation of sentence structures, the order in which sentences of these structures are deployed, the points where the student’s name is invoked, and the fact that all the sentences describe the students with the exception of the final sentence, which is delivered as a word of encouragement addressed directly to the student. Vary sentences widely. 

Finally, note that the user may opt to run this bot in 21CC mode – in which the responses should draw first and foremost on Singapore’s Ministry of Education’s 21st Century Competencies and a set of values known as the R3ICH values – or in BLGPS mode, specific to Boon Lay Garden Primary School, in which case the responses draw on the 21st Century Competencies, R3ICH values as well as a set of terms and values known as the BLGPS Student Outcomes. 
Assuming that this entire prompt has been input in the backend or has been pasted in this chat, you are to move forward in the same way, with all input from here kept in the context window. In order to best construct your output, I will let you know how a user will structure their input, and how to interpret this input so as to produce remarks that are of the requisite quality, length, and nature. We will begin with your output guide. 

In this entire prompt, the word "student" in parentheses like so (student) is to be used in lieu of names. However, in your output, use the index number provided in the input (e.g. 01, 02) as the identifier for each student, and construct based on the parameters below.
For every student, the prompts I provide will take the form of a string of information. This string of information will contain the student's name, descriptors/descriptors that apply to the student, and a score from 1 to 5. If users ask, let them know that the score here is mapped broadly to the student's conduct grade, with 5 corresponding to the "Excellent" grade. Start the comment with the name. Use the descriptors as direction for crafting the comment. Use the score of 1-5 to estimate how positive and effusive to be, with scores of 5 warranting the most positive comments. Before each set of prompts delivered in this manner, I will let you know the pronouns of the students that we are working on, then send the prompts for all the students who use the same pronouns. For instance, a prompt beginning with “she/her” followed by a long string of student data would require all responses pertaining to that data to use the “she/her” pronouns. 

Additionally, remember to map your output to the relevant reference points based on whether you are in BLGPS mode or 21CC mode. If not specified, assume that you are in BLGPS mode. The following paragraphs contain information for how best to map values. 

Value mapping for 21CC mode: In 21CC mode, whatever the user input, first check whether there is any potential for alignment with the following. 
You may draw from MOE Singapore’s 21st Century Competencies: 
a. Critical thinking 
b. Adaptive thinking 
c. Inventive thinking 
d. Communication skills
e. Collaboration skills
f. Information skills
g. Civic literacy
h. Global literacy
i. Cross-cultural literacy 
If a student displays any of the characteristics on the list based on the user input, always include at least one of these 21st Century Competencies. 

Further, you may draw from the following list of values, knows as R3ICH values:
a. Respect
b. Responsibility
c. Resilience 
d. Integrity
e. Care
f. Harmony
Finally, you may draw from the following list of social-emotional competencies:
a. Self-awareness
b. Self-management
c. Responsible decision making
d. Social awareness
e. Relationship management
Any input adjective from the user which can be seen to be indicative of the student in question possessing or exhibiting qualities in the lists above allows you to then use the relevant qualities in the list to expand your description. This is true regardless of whether you are in BLGPS mode or 21CC mode.

For BLGPS mode, please refer to the definitions of CC CL CC for the value mapping. The definitions are: 
Curious thinker: 
A curious thinker displays a lively curiosity and desire to learn through asking insightful and important questions. They also exercise critical thinking with sound reasoning and decision-making; and metacognition. Further, they show adaptive thinking by assessing different contexts and adjusting perspectives to manage complexities. Finally, they demonstrate innovative thinking by generating novel and useful ideas, then evaluate and refine ideas to formulate solutions. Some relevant additional descriptors which can be used to describe a curious thinker are: inquisitive, diligent, bright, meticulous, sharp, quick-witted, curious, asks questions, participates actively, hardworking.
If the descriptors in the prompt at all relate to anything under the curious thinker label, make liberal reference to this list of descriptors and the profile of a curious thinker in the paragraph above, regardless of whether the user sets you in BLGPS mode or not, but do not use the category title itself. 

Confident learner: 
A confident learner shows motivation to learn, exhibits self-directedness when learning, displays resilience to overcome challenges in learning and demonstrates connectedness to learn well with others by using effective communication and collaborative skills. Some relevant additional descriptors which can be used to describe a confident learner are: hardworking, personable, outgoing, responsible, good leader, kind, helpful, responsible, confident, self-directed, self-motivated.
If the descriptors in the prompt at all relate to anything under the confident learner label, make liberal reference to this list of descriptors and the profile of a confident learner in the paragraph above, regardless of whether the user sets you in BLGPS mode or not, but do not use the category title itself.

Compassionate contributor: 
A compassionate contributor works cooperatively with harmony, adopts an attitude of care and integrity in daily interactions, demonstrates respect in communication and action, puts in effort through responsibility and resilience to contribute to the community, and shows civic, global and cross-cultural awareness. Some relevant additional descriptors which can be used to describe a compassionate contributor are: team-player, cheerful, personable, well-liked, kind, caring.
If the descriptors in the prompt at all relate to anything under the compassionate contributor label, make liberal reference to this list of descriptors and the profile of a compassionate contributor in the paragraph above, regardless of whether the user sets you in BLGPS mode or not, but do not use the category title itself. 

Regardless of whether you are in BLGPS mode or 21CC mode, you need not take the descriptors exactly as delivered; find the best fit for at least one of the provided descriptors and use the 21st Century Competency framework to express it instead. Treat the above list as a word bank and make heavy use of the phrases available at every opportunity. 

When the descriptors are not completely positive, frame the comment to the effect of “They can do even better if_____________.” We will always maintain a positive, upbeat tone and adopt a growth mindset. 
DO NOT USE THE CT CC CL FRAMEWORK unless you are explicitly set to BLGPS mode. 

In certain cases, the descriptors may be followed by parentheses which provide additional information. Incorporate these in the 2nd or 3rd sentence. Do not mention them in the 1st sentence.

Responses should vary sentence structure actively and ensure that phrasing is positive and upbeat. Only the very last line of the comment should be talking to the student directly. Responses should not be fewer than 55 words, and can be up to 75 words. 

Required Footer:
- If 21CC mode: **Note that the output from here is a first draft, and will always require editing as the AI is not capable of producing flawlessly accurate output. You may consider using the "Italicize Inaccuracies" button in the sidebar to kickstart your editing process.**
- If BLGPS mode: **Note that the output from here is a first draft, and will always require editing as the AI is not capable of producing flawlessly accurate output. Pay special attention to whether the descriptors match the BLGPS student outcome mentioned in the remark, if any. You may consider using the "Italicize Inaccuracies" button in the sidebar to kickstart your editing process.**

Available actions (accessible via sidebar buttons — the user does NOT type these as commands):
- "Guide / Instructions" button: When the user triggers this, display a clear usage guide explaining:
  1. Input format: start with pronouns (she/her or he/him), then for each student provide their index number, descriptors (space-separated), optional parenthetical info, and a conduct score 1-5.
  2. Example: she/her 01 responsible hardworking (Class Monitor) 5, 02 cheerful friendly 4
  3. The two modes available: BLGPS Mode and 21CC Mode, selectable in the sidebar.
  4. After generating, use the sidebar buttons: "Analyse Class Data" for a competency breakdown table, and "Italicize Inaccuracies" to highlight areas that may need editing.
  5. The app also supports a "Names Enabled" input mode where teachers can enter student names in a structured form — names are kept private and never sent to the AI.
- "Analyse Class Data" button: Provide a breakdown of competencies/values used in the remarks in table format.
- "Italicize Inaccuracies" button: Repeat the remarks verbatim but italicize areas that are likely candidates for inaccuracy or that the teacher should double-check.
"""

# --- FEW-SHOT EXEMPLARS (fed as conversation history) ---
EXEMPLARS = [
    ("she/her 01 softspoken compassionate 4", "01 is a soft-spoken and kind-hearted student. Possessing the traits of a compassionate contributor, she consistently shows care and empathy in her interactions with her peers. Her gentle disposition and sincerity are admirable qualities that create a positive and harmonious classroom environment. She is a valued member of the class. Keep up the wonderful attitude, 01!"),
    ("she/her 02 cheerful participative (can be more consistent in attendance) 4", "A cheerful and participative student, 02 brings a great deal of positive energy to the classroom. She engages readily in discussions and group activities, demonstrating the qualities of a confident learner. Her contributions are always valued by her peers and teachers alike. She is encouraged to maintain consistent attendance to maximise her learning opportunities. You are on the right track, 02!"),
    ("she/her 03 responsible compassionate considerate observant 5", "03 is a responsible and compassionate student who demonstrates a remarkable sense of awareness. As a compassionate contributor, she is considerate of the needs of others and is always observant, often being the first to offer assistance. Her integrity and strong sense of responsibility make her a trusted and respected role model in class. We are very proud of you, 03!"),
    ("she/her 04 easygoing helpful participative 4", "An easygoing and helpful student, 04 is a wonderful team player. She gets along well with her peers and participates actively in class, exemplifying the collaborative spirit of a compassionate contributor. Her willingness to lend a hand and her positive attitude significantly enhance the learning environment for everyone around her. Good work, 04!"),
    ("she/her 05 diligent bright driven 5", "05 is a diligent and bright student with a remarkable drive to succeed. A curious thinker, she approaches her work with meticulous care and is consistently self-motivated to produce work of the highest quality. Her inquisitive nature and inventive thinking allow her to grasp complex concepts with ease and to excel in her studies. Well done, 05!"),
    ("she/her 06 responsible dependable 4", "A responsible and dependable student, 06 can always be counted on to complete her tasks with care and diligence. She displays a high degree of self-management and takes ownership of her learning. This sense of responsibility is a key trait of a confident learner and serves her well in all her academic pursuits. Keep it up, 06!"),
    ("she/her 07 holds herself to a high standard considerate model student 4", "07 is a considerate and conscientious student who consistently holds herself to a high standard. She truly exemplifies what it means to be a confident learner, showing resilience and a desire for excellence in all that she does. Her exemplary behaviour and thoughtful nature make her a positive influence and a model student for her peers. Continue to work hard, 07!"),
    ("she/her 08 hardworking sincere reliable 4", "08 is a hardworking, sincere, and reliable student. She approaches all her responsibilities with integrity and a positive attitude, making her a trustworthy member of the class. Possessing the traits of a confident learner, she perseveres through challenges and consistently puts forth her best effort in her studies. Excellent work, 08!"),
    ("she/her 09 hardworking dependable (quiet) 4", "09 is a hardworking and dependable student who takes great pride in her work. Though she has a quiet disposition, her focus and diligence are evident in the high quality of her assignments. As a confident learner, she demonstrates resilience and a strong sense of responsibility. We are proud of her consistent efforts. You can do it, 09!"),
    ("she/her 10 respectful participative hardworking 4", "A respectful and hardworking student, 10 is an active participant in the classroom. She shows great respect for her peers and teachers and contributes thoughtfully during class discussions. Her dedication and collaborative spirit are the hallmarks of a confident learner, and she is always willing to put in the effort to achieve her goals. Keep up the good work, 10!"),
    ("she/her 11 cheerful friendly confident 4", "11 is a cheerful, friendly, and confident student. Her positive disposition and ability to manage relationships effectively make her a well-liked member of the class. As a confident learner, she is not afraid to take on new challenges and actively engages with her learning. Her bright personality is a joy to have in class. Well done, 11!"),
    ("she/her 12 softspoken hardworking 4", "12 is a soft-spoken and hardworking student who demonstrates great focus in her studies. A confident learner, she is diligent and meticulous in completing her assignments, always striving to do her best. Her quiet determination and resilience are admirable qualities that will undoubtedly lead her to continued success in her learning journey. Keep it up, 12!"),
    ("she/her 13 resilient well-liked compassionate 4", "13 is a resilient and compassionate student who is well-liked by her peers. She exemplifies the qualities of a compassionate contributor, showing care for others and working harmoniously in group settings. She faces challenges with a positive attitude and perseveres, demonstrating a strong sense of self-management and determination. Good work, 13!"),
    ("she/her 14 softspoken sincere kind 4", "A soft-spoken and sincere student, 14 is exceptionally kind to everyone she interacts with. Her gentle and caring nature reflects the core values of a compassionate contributor. She demonstrates respect and integrity in her actions, creating a warm and supportive atmosphere around her. Her thoughtful presence is deeply appreciated in the classroom. Excellent work, 14!"),
    ("she/her 15 reliable hardworking 4", "15 is a reliable and hardworking student who can be depended upon to give her best effort. She approaches her studies with a responsible attitude, consistently completing her work to a high standard. This display of diligence and self-management are key characteristics of a confident learner, setting her up for continued progress. Keep it up, 15!"),
    ("she/her 16 dependable reserved (speak up more) 4", "16 is a dependable student with a thoughtful and reserved nature. Her work is always completed with care, showing a strong sense of responsibility. To further develop her communication skills as a confident learner, she is encouraged to share her insightful ideas more frequently during class discussions. Her perspective is valuable. You can do it, 16!"),
    ("she/her 17 cheerful resilient confident 4", "17 is a cheerful and resilient student who exudes confidence. A confident learner, she embraces challenges with a positive mindset and is not discouraged by setbacks. Her ability to self-manage and persevere is commendable. Her bright and determined spirit makes her a wonderful and inspiring presence in the classroom. Well done, 17!"),
    ("she/her 18 dependable model student driven sincere 5", "18 truly exemplifies what it means to be a model student. She is a driven and sincere individual who is exceptionally dependable. As a confident learner, her integrity and commitment to excellence are evident in all she does, inspiring her peers to also strive for their best. She consistently demonstrates responsibility and a strong work ethic. We are proud of you, 18!"),
    ("she/her 19 compassionate caring hardworking 5", "19 is a compassionate and hardworking student. Possessing the qualities of a compassionate contributor, her caring nature is evident in her daily interactions, where she is always ready to offer help and support to her peers. This empathy, combined with her diligent approach to her studies, makes her a truly well-rounded and admirable individual. Excellent work, 19!"),
    ("he/him 20 reliable dependable hardworking 4", "20 is a reliable and dependable student who approaches his studies with a commendable work ethic. He takes responsibility for his learning and can always be trusted to complete his tasks diligently. His consistency and commitment are the hallmarks of a confident learner, and these traits will serve him well as he continues to progress. Keep up the great work, 20!"),
    ("he/him 22 friendly cheerful (can focus better) 4", "A friendly and cheerful student, 22 brings a great deal of positive energy to the classroom. His outgoing nature allows him to interact well with his peers. He is a compassionate contributor who adds to the harmony of the class. He has the potential to achieve even more by channeling his enthusiasm into a sustained focus during lessons. You can do it, 22!"),
    ("he/him 23 trustworthy reliable dependable helpful sincere 5", "23 is an exceptionally trustworthy and reliable student. His integrity is beyond reproach, and he is a sincere and helpful member of the class. As a compassionate contributor, he can always be depended upon to assist his peers and teachers, demonstrating a strong sense of responsibility and care for his community. It is a joy to have him in class. Well done, 23!"),
    ("he/him 24 easygoing friendly (could work better with others) 3", "An easygoing and friendly student, 24 gets along well with his classmates on a personal level. He brings a pleasant and calm demeanor to the classroom. To grow further as a compassionate contributor, he is encouraged to develop his collaboration skills and work more effectively within a team setting to achieve shared goals. You are on the right track, 24!"),
    ("he/him 25 bright outspoken (can work harder) 3", "25 is a bright and outspoken student who is confident in sharing his ideas. His willingness to speak up shows great potential for communication and leadership. He can unlock his full capabilities as a curious thinker by applying himself more consistently to his studies and channelling his intelligence into diligent work. Continue to work hard, 25!"),
    ("he/him 27 hardworking considerate (could focus better) 4", "27 is a hardworking and considerate student who is always mindful of others. As a compassionate contributor, his thoughtfulness is greatly appreciated by his peers. He shows diligence in his work and is capable of producing excellent results. By improving his ability to focus during lessons, he will be able to perform even better. Keep it up, 27!"),
    ("he/him 28 reliable easygoing helpful 5", "28 is a reliable, easygoing, and helpful student. He is a shining example of a compassionate contributor, always willing to lend a hand to his teachers and peers without hesitation. His responsible and caring attitude contributes significantly to a positive and harmonious classroom environment, making him a valued member of the class. Excellent work, 28!"),
    ("he/him 29 hardworking driven sincere 5", "29 is a hardworking and driven student who approaches his studies with sincerity and focus. A confident learner, he is self-motivated and determined to achieve his best, demonstrating great resilience and integrity in his work. His strong work ethic and commitment to personal growth are truly admirable qualities that set a fine example for others. Well done, 29!"),
    ("he/him 30 respectful disciplined (could speak up more) 4", "30 is a respectful and disciplined student whose behaviour is commendable. He demonstrates a high degree of self-management in his actions and conduct. To continue his development as a confident learner, he is encouraged to build his confidence and share his valuable thoughts more often during class, as his perspective would enrich discussions. You can do it, 30!"),
    ("he/him 31 helpful sincere (could work better with others) 4", "31 is a helpful and sincere student with a kind disposition. He demonstrates the qualities of a compassionate contributor through his willingness to assist others. He stands to grow even more by enhancing his collaboration skills and learning to work more cohesively with his peers in group settings to achieve common objectives. Keep up the effort, 31!"),
    ("he/him 32 outspoken cheerful (could focus better) 3", "An outspoken and cheerful student, 32 is never hesitant to share his thoughts. His energy is a positive asset in the classroom. He is encouraged to practice greater self-management by focusing his attention more consistently during instruction, which will help him better harness his natural enthusiasm for learning and achieve greater success. You are on the right track, 32!"),
    ("he/him 33 insightful inquisitive (could work better with others) 3", "33 is an insightful and inquisitive student who thinks deeply about his learning. Possessing the traits of a curious thinker, he often has unique perspectives to offer. He can further develop his collaboration and communication skills by learning to integrate his ideas more effectively with those of his teammates during group work. Keep working at it, 33!"),
    ("he/him 34 reliable good leader compassionate 5", "34 is a reliable and compassionate student who has demonstrated excellent leadership qualities. A confident learner, he takes on responsibilities with maturity and guides his peers with care and respect. His ability to lead with empathy makes him an effective and trusted leader, embodying the spirit of a compassionate contributor. We are very proud of you, 34!"),
    ("he/him 35 respectful dependable (could work on confidence) 3", "A respectful and dependable student, 35 consistently shows a responsible attitude towards his duties. His polite and reliable nature is appreciated by all. To continue his growth as a confident learner, he is encouraged to believe more in his own abilities and step out of his comfort zone, as this will help build his self-confidence. You can do it, 35!"),
    ("he/him 36 hardworking softspoken reliable 5", "36 is a hardworking and reliable student. A confident learner, he completes all his work with diligence and a quiet determination that is truly admirable. Though he is soft-spoken, his commitment and strong sense of responsibility are evident in the consistent high quality of his work. His efforts are a shining example to his peers. Excellent work, 36!"),
    ("he/him 37 outspoken distracted (could work better with others) 3", "37 is an outspoken student with a great deal of energy. His willingness to voice his opinion is a strength. To grow as a learner, he would benefit from channelling his energy into focused tasks and developing his collaboration skills. By learning to work more harmoniously and effectively with his peers, he will make more positive contributions. Keep trying, 37!"),
    ("he/him 38 respectful hardworking dependable 4", "38 is a respectful, hardworking, and dependable student. He consistently demonstrates the core values of responsibility and resilience in his approach to his studies. As a confident learner, he can be counted on to complete his work diligently and to the best of his ability, making him a reliable and trusted member of the class. Good work, 38!"),
    ("he/him 39 cheerful playful outspoken (could focus better) 3", "39 is a cheerful and playful student who is confident and outspoken. He brings a lot of vibrant energy to the class. His next step in his development is to practice self-management by improving his focus during lessons. By doing so, he can better apply his bright and energetic personality towards his academic goals. You are on the right track, 39!"),
    ("she/her 40 easygoing friendly participative 4", "40 is an easygoing and friendly student who participates actively in class. Her approachable nature makes her a wonderful team member, and she embodies the spirit of a compassionate contributor. She communicates well with her peers and her willingness to contribute to discussions helps to create a dynamic and collaborative learning environment. Keep up the good work, 40!"),
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
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(temperature=0.4)
        )
        chat = model.start_chat(history=_build_history(current_mode))
        formatted_input = f"CURRENT MODE: {current_mode}\n\nUSER INPUT: {user_text}"
        response = chat.send_message(formatted_input)
        text = response.text
        # Ensure the footer note is clearly separated by a paragraph break
        text = re.sub(r'(?<!\n)\n(\*{0,2}Note that)', r'\n\n\1', text)
        return text
    except Exception as e:
        return f"⚠️ API Error: {e}"

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
    h2, h3 { color: #2dd4bf !important; }
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
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "accuracy_result" not in st.session_state:
    st.session_state.accuracy_result = ""

st.markdown("<h1 style='color: #f59e0b !important;'>🇸🇬 Student Remarks Assistant</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode_selection = st.radio("Target Mode:", ["BLGPS Mode", "21CC Mode"])
    st.divider()
    st.subheader("Special Actions")
    help_clicked = st.button("📖 Guide / Instructions")
    analyse_clicked = st.button("📊 Analyse Class Data")
    accuracy_clicked = st.button("🔍 Italicize Inaccuracies")

# --- INPUT AREA (Tabs) ---
tab_quick, tab_names = st.tabs(["⚡ Quick Entry", "📝 Names Enabled"])

with tab_quick:
    st.button("📋 Load Sample Data", on_click=_load_sample_quick, key="sample_quick")
    user_data_input = st.text_area("Input Student Details:",
                                   placeholder="she/her 01 responsible (Class Monitor) 5, 02 helpful 4",
                                   height=150, key="quick_input")
    if st.button("🚀 Generate Remarks", type="primary", key="gen_quick"):
        if user_data_input:
            with st.spinner("Generating..."):
                res = call_gemini(user_data_input, mode_selection)
                st.session_state.last_remarks = res
                st.session_state.last_input = user_data_input
                st.session_state.name_map = {}
                st.session_state.analysis_result = ""
                st.session_state.accuracy_result = ""
            st.toast("✅ Remarks generated!")
        else:
            st.error("Please enter student data.")

with tab_names:
    st.caption("Student names are kept private — only index placeholders are sent to the AI.")
    st.button("📋 Load Sample Data", on_click=_load_sample_names, key="sample_names")
    pronouns = st.radio("Pronouns for this batch:", ["she/her", "he/him"], horizontal=True, key="pronouns_radio")
    num_students = st.number_input("Number of students:", min_value=1, max_value=45, value=1, step=1, key="num_students_input")
    st.caption("Rating guide: 5 = Excellent · 4 = Very Good · 3 = Good · 2 = Satisfactory · 1 = Needs Improvement")

    students = []
    for i in range(int(num_students)):
        with st.expander(f"Student {i + 1}", expanded=(i == 0)):
            name = st.text_input("Name", key=f"name_{i}", placeholder="e.g. Wei Lin")
            characteristics = st.text_input("Characteristics / Descriptors", key=f"chars_{i}",
                                            placeholder="e.g. cheerful, responsible, hardworking")
            roles = st.text_input("Class / Leadership Roles (optional)", key=f"roles_{i}",
                                  placeholder="e.g. Class Monitor, Prefect")
            awards = st.text_input("Awards (optional)", key=f"awards_{i}",
                                   placeholder="e.g. Good Progress Award")
            other = st.text_input("Other Information (optional)", key=f"other_{i}",
                                  placeholder="e.g. frequent latecoming, can focus better")
            rating = st.selectbox("Behaviour Rating", options=[5, 4, 3, 2, 1], key=f"rating_{i}",
                                  help="5 = Excellent conduct, 1 = Needs significant improvement")
            students.append({
                "name": name.strip(),
                "characteristics": characteristics.strip(),
                "roles": roles.strip(),
                "awards": awards.strip(),
                "other": other.strip(),
                "rating": rating,
            })

    if st.button("🚀 Generate Remarks", type="primary", key="gen_names"):
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
                st.session_state.analysis_result = ""
                st.session_state.accuracy_result = ""
            st.toast("✅ Remarks generated!")

# --- PERSISTENT OUTPUT ---
if st.session_state.last_remarks:
    st.divider()
    st.markdown("### Generated Remarks")
    st.write(st.session_state.last_remarks)
    txt_data = _remarks_to_txt(st.session_state.last_remarks)
    st.download_button("📥 Download TXT", txt_data, file_name="student_remarks.txt", mime="text/plain")
    with st.expander("📋 Copy raw text"):
        st.code(st.session_state.last_remarks, language=None)

# Process Sidebar Actions
if help_clicked:
    with st.spinner("Loading guide..."):
        st.info(call_gemini("Display the full usage guide for this app, explaining the input format, modes, and available sidebar buttons.", mode_selection))

if analyse_clicked:
    if st.session_state.last_remarks:
        with st.spinner("Analysing..."):
            sanitized_remarks = st.session_state.last_remarks
            if st.session_state.name_map:
                for idx, name in st.session_state.name_map.items():
                    sanitized_remarks = sanitized_remarks.replace(name, idx)
            context = f"Here are the remarks I previously generated based on this input:\n\nINPUT: {st.session_state.last_input}\n\nREMARKS:\n{sanitized_remarks}\n\nNow analyse these remarks: provide a breakdown of competencies/values in table format."
            res = call_gemini(context, mode_selection)
            if st.session_state.name_map:
                res = _restore_names(res, st.session_state.name_map)
            st.session_state.analysis_result = res
    else:
        st.warning("Generate remarks first before analysing.")

if accuracy_clicked:
    if st.session_state.last_remarks:
        with st.spinner("Checking accuracy..."):
            sanitized_remarks = st.session_state.last_remarks
            if st.session_state.name_map:
                for idx, name in st.session_state.name_map.items():
                    sanitized_remarks = sanitized_remarks.replace(name, idx)
            context = f"Here are the remarks I previously generated based on this input:\n\nINPUT: {st.session_state.last_input}\n\nREMARKS:\n{sanitized_remarks}\n\nNow repeat the remarks verbatim but italicize areas likely for inaccuracy."
            res = call_gemini(context, mode_selection)
            if st.session_state.name_map:
                res = _restore_names(res, st.session_state.name_map)
            st.session_state.accuracy_result = res
    else:
        st.warning("Generate remarks first before checking accuracy.")

if st.session_state.analysis_result:
    st.divider()
    st.markdown("### 📊 Class Data Analysis")
    st.write(st.session_state.analysis_result)

if st.session_state.accuracy_result:
    st.divider()
    st.markdown("### 🔍 Accuracy Check")
    st.write(st.session_state.accuracy_result)

st.divider()
st.caption("Powered by Gemini 2.5 Flash | Framework: Singapore MOE 21CC / BLGPS")


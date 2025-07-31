from reelsmith.stub import State, ImagePromptSegment, ScriptWords, ImageSegmentList
from reelsmith.llm import LLM


class ScriptGenerator:
    def __init__(self, llm: LLM, instruction: str):
        self.llm = llm
        self.instruction = instruction

    def generate_script_words(self, state: State) -> State:
        prompt = """
You are a helpful assistant that writes short, coherent video scripts for narration.
You will be given a script topic, and research performed on that topic by surfing the web. You have to generate a script 
of 12–15 sentences that explains or tells a story about the topic in clear and natural language.

Return the script as a JSON object with:
- script_words: a list of strings, where each item is a word or punctuation mark (e.g., ".", ",") in the order they appear 
in the script.

Rules:
- Punctuation (like ".", ",", "!") should be separate items in the list.
- Do not include any metadata, markdown, or formatting.
- The script should be grammatically correct and easy to narrate.

Now generate the script_words list for the following topic:

Topic: {topic}
Research: {research}
        """

        state.script_words = self.llm.invoke(prompt.format(topic=state.topic, research=state.search_summary),
                                             output_structure=ScriptWords)

        return state

    def generate_image_prompts(self, state: State) -> State:
        prompt = """
You are a creative assistant generating image prompts for an AI-generated video narration.

You will be given a list of words (`script_words`) that form a narration script. Your task is to divide the script into 6–8 coherent, meaningful segments and generate a visually rich and descriptive image prompt for each one.

Your response must be a JSON object with:
- image_segments: a list of objects. Each object contains:
  - prompt: a detailed, vivid natural language image prompt that can be used by AI models like DALL·E or Midjourney.
  - word_range: a tuple [start_index, end_index] indicating the **inclusive** range of script_words that the image is based on.

The output **must** follow these strict constraints:
- The first word_range must **start at index 0**
- The last word_range must **end at index {last_index}**
- The ranges must be **continuous** and **non-overlapping**: each word index from 0 to {last_index} must appear in exactly one word_range
- The number of segments must be 6, 7, or 8.

The prompt should be **very specific, detailed, and grounded in the actual content** of the corresponding script_words slice. Avoid vague prompts.

Examples:
If the script_words segment describes a historical event, a good prompt would be:  
  "A dusty battlefield in 1860s America, Union and Confederate soldiers in blue and grey uniforms clashing under a smoky sky"

If the script describes future technology:  
  "A sleek, autonomous flying car hovering over a neon-lit smart city at night, glowing holograms and glass skyscrapers in the background"

If it talks about a biological process:  
  "A close-up of white blood cells attacking virus particles inside the human bloodstream, with red blood cells in the background"

If these rules are not strictly followed, your response will be considered invalid and discarded.

Here are the script_words to process:  
{script_words}
        """

        state.image_segments = self.llm.invoke(
            prompt.format(length=len(state.script_words.script_words), script_words=state.script_words.script_words,
                          last_index=len(state.script_words.script_words) - 1),
            output_structure=ImageSegmentList)

        return state

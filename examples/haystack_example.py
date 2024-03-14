import pandas as pd
from haystack.components.generators import (
    HuggingFaceLocalGenerator,
    OpenAIGenerator,
)
from haystack.utils import Secret

from annomatic.annotator import FileAnnotator
from annomatic.prompt import Prompt

# create dataframe with textual data
text = [
    "Recent studies suggest that the new technology is revolutionary.",
    "Critics argue that the government's policies are misguided and harmful.",
]
df = pd.DataFrame({"text": text})

# create a Prompt object (with f-string template)
prompt = Prompt()
prompt.add_part("Instruction: '{text}'")
prompt.add_labels_part("Classify the sentence above as {label}.")
prompt.add_part("Output: ")

# generator = OpenAIGenerator(
#    api_key=Secret.from_token(
#        token="KEY"))

generator = HuggingFaceLocalGenerator(
    task="text2text-generation",
    generation_kwargs={"max_new_tokens": 100, "temperature": 0.9},
)

annotator = FileAnnotator(
    model=generator,
    out_path="./output.csv",
    out_format="csv",
)

annotator.set_data(df, data_variable="text")
annotator.set_prompt(prompt)

# annotate the data and return as df
result = annotator.annotate(label=["BIASED", "NOT BIASED"], return_df=True)
print(result)

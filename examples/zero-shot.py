import pandas as pd
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator

from annomatic.annotator import FileAnnotator

# create dataframe with textual data
text = [
    "Recent studies suggest that the new technology is revolutionary.",
    "Critics argue that the government's policies are misguided and harmful.",
]
df = pd.DataFrame({"text": text})

prompt = PromptBuilder(
    "Instruction: '{{text}}'\n"
    "Classify the sentence above as BIASED or NOT BIASED.\n"
    "Output: ",
)

# create model via Haystack 2.0
model = HuggingFaceLocalGenerator(
    task="text2text-generation",
    generation_kwargs={"max_new_tokens": 100, "temperature": 0.9},
)

# create annotator
annotator = FileAnnotator(
    model=model,
    out_path="./output.csv",
    out_format="csv",
    batch_size=2,
    labels=["BIASED", "NOT BIASED"],
)

annotator.set_data(df, data_variable="text")
annotator.set_prompt(prompt)

# annotate the data and return as df
result = annotator.annotate(return_df=True)
print(result)

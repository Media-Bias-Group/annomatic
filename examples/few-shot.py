import pandas as pd
from haystack.components.generators import HuggingFaceLocalGenerator

from annomatic.annotator import FileAnnotator
from annomatic.prompt import Prompt

# provide 2 examples of biased and not biased sentences
data = {
    "text": [
        "Recent studies suggest that the new technology is revolutionary.",
        "Critics argue that the government's policies are misguided "
        "and harmful.",
    ],
    "label": [
        "NOT BIASED",
        "BIASED",
    ],
}
df_examples = pd.DataFrame(data, columns=["text", "label"])

# sentence to be annotated
query = ["The new technology is revolutionary."]
df = pd.DataFrame({"text": query})

# create a Prompt object (with f-string template)
prompt = Prompt()
prompt.add_part("Instruction: '{text}'")
prompt.add_labels_part("Classify the sentence above as {label}.")
prompt.add_part("Output: ")

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
)

# set data and prompt
annotator.set_data(df, data_variable="text")
annotator.set_prompt(prompt)

annotator.set_context(df_examples)

# annotate the data and return as df
result = annotator.annotate(label=["BIASED", "NOT BIASED"], return_df=True)
print(result)

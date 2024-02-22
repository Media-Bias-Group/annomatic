import pandas as pd

from annomatic.annotator import HuggingFaceFileAnnotator
from annomatic.config.base import HuggingFaceConfig
from annomatic.config.factory import ConfigFactory
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

# define HuggingFace config with
config: HuggingFaceConfig = ConfigFactory.create(
    "huggingface",
)

# create HuggingFace annotator and set data and prompt
annotator = HuggingFaceFileAnnotator(
    model_name="google/flan-t5-base",
    config=config,
    out_path="./output.csv",
    auto_model="AutoModelForSeq2SeqLM",
)

annotator.set_data(df, data_variable="text")
annotator.set_prompt(prompt)

# uses the regular prompt and prints the label as suffix.
annotator.set_context(df_examples)

# annotate the data and return as df
result = annotator.annotate(label=["BIASED", "NOT BIASED"], return_df=True)
print(result)

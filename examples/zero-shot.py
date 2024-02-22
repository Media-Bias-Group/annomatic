import pandas as pd

from annomatic.annotator import HuggingFaceFileAnnotator
from annomatic.config.base import HuggingFaceConfig
from annomatic.config.factory import ConfigFactory
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

# define HuggingFace config with
config = ConfigFactory.create(
    "huggingface",
    pad_token_id=50256,
)
config.tokenizer_args["pad_token_id"] = 50256
config.tokenizer_args["padding_side"] = "left"

# create HuggingFace annotator and set data and prompt
annotator = HuggingFaceFileAnnotator(
    model_name="EleutherAI/gpt-neo-1.3B",
    config=config,
    out_path="./output.csv",
)

annotator.set_data(df, data_variable="text")
annotator.set_prompt(prompt)

# annotate the data and return as df
result = annotator.annotate(label=["BIASED", "NOT BIASED"], return_df=True)
print(result)

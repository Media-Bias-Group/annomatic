import pandas as pd
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator

from annomatic.annotator import FileAnnotator

# provide 2 examples of biased and not biased sentences
df_examples = pd.DataFrame(
    {
        "text": [
            "Recent studies suggest that the new technology is revolutionary.",
            "Critics argue that the government's policies are misguided "
            "and harmful.",
        ],
        "label": [
            "NOT BIASED",
            "BIASED",
        ],
    },
)

# sentence to be annotated
query = ["The new technology is revolutionary."]
df = pd.DataFrame({"text": query})

prompt = PromptBuilder(
    "{% for key,value in examples.iterrows() %}"
    "Instruction: '{{value.text}}'\n"
    "Classify the sentence above as BIASED or NOT BIASED.\n"
    "Output: {{ value.label }}\n\n"
    "{% endfor %}"
    "Instruction: '{{text}}'\n"
    "Classify the sentence above as BIASED or NOT BIASED.\n"
    "Output: ",
)

# create Haystack 2.0 model
model = HuggingFaceLocalGenerator(
    task="text2text-generation",
    generation_kwargs={"max_new_tokens": 100, "temperature": 0.9},
)

# create annotator
annotator = FileAnnotator(
    model=model,
    out_path="./output.csv",
    out_format="csv",
    labels=["BIASED", "NOT BIASED"],
)

# set data and prompt
annotator.set_data(df, data_variable="text")
annotator.set_prompt(prompt)

# map the examples to the prompt value for examples
annotator.set_context({"examples": df_examples})

# annotate the data and return as df
result = annotator.annotate(return_df=True)
print(result)

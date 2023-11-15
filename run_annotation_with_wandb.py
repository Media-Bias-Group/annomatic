from datasets import load_dataset
import pandas as pd
from annomatic.prompt import Prompt
from annomatic.annotator import HuggingFaceCsvAnnotator
import wandb

# login to wandb
wandb.login(key='')
project_name = 'annomatic'


# load babe
dataset = load_dataset('mediabiasgroup/BABE-v3')
df_babe = pd.DataFrame(dataset['train'])


prompt = Prompt()
prompt.add_part("Instruction: '{text}'")
prompt.add_labels_part("Classify the sentence above as {label}.")
prompt.add_part("Output: ")


model_name = "NousResearch/Llama-2-7b-chat-hf"

# split the modelname from '/'
parts = model_name.split('/')
if len(parts) == 2:
    output_name = parts[1]
else:
    output_name = parts[0]

run = wandb.init(project=project_name,name=output_name)


model_args = {'device_map': 'auto', 'load_in_8bit': True, }
annotator = HuggingFaceCsvAnnotator(model_name=model_name,
                                     out_path=f"./{output_name}.csv",
                                     model_args=model_args)


annotator.set_prompt(prompt)
annotator.set_data(data=df_babe, in_col="text")


# define a specific batch_size (default=5)
annotator.batch_size = 10
annotator.annotate(label=["BIASED","NOT BIASED"])


artifact = wandb.Artifact(
            name=output_name,
            type="dataset",
        )
artifact.add_file(local_path=f"./{output_name}.csv", name=output_name)
run.log_artifact(artifact)
wandb.finish()

"""
I know that it time. but it is my next step.
My idea is to make control from a single file.
Just try my friend I don't blame you. Don't be discourage by time.
Just enjoy the process and do what you can. Nothing more, to not poison mind.
I run main script. It checks progress and store hierarchy.
Btw I should do exit via main, which will ask me for some prompt which I will record in console, store it under my notes and after I finished type do exit.
Each file should have id, each file should be linked. if it entry point then [*, fileName] to point it is special. 
I don't care if it useless, it is nice challenge.
If file produce some product, then product id is family_id plus indivudual_id. if program needs, then it should address primary to family_id.
Let's begin....
well let's try...
I recall, main do in steps... It checks status of project... if script done its job, then the step is completed and recorded (I suppose in must case, there must be product
of the script thus it checked by existence of product). during new run, I can create history of product knowing each from each is growing, separate entity for production, here 
I will use full id for products which is family_id and individual_id.
Maybe I should have journal of intentions.
How I should manage scripts id?.... what for example if I transfered it... then I should search across all project all files to find with such id... but it will take long, then confirmation
before doing so.
is it useless as I add to much adherent? maybe but main pipeline must be.... because I fear, I don't want to touch it.
Well let's start.
I start the project with collecting data. I make folder data_collection, here is file api_data_collector.py. Sorry myself for unaccurate action but here
what it produce and use: 
base_url = 'https://jyx.jyu.fi'
collection_url = f'{base_url}/rest/collections/3e5cf62e-86f2-44c0-beed-88ff6836d55b/items'
output_file = 'data_collection/output.json'
state_file = 'data_collection/state.json'
So continue to thinking...
base_url and collection_url are aka arguments from where to collect, it is assumed that I want to get all till it dry.
in state.json... it used by script itself and it is product... it defines the size of chunks to retrieve data and saves result where it stopped
in current state it not adapted to be called, values is hardcoded as argument inside code and no options to pass them currently.
It assumes to work with only one source...
output.json is essential product of the script which will be connection point in chain to other.
Next step is data preparation, scripts locates in data_preparation and they mainly utilize output.json
what a garbage...just continue writing. because you don't have other option, and I suppose it somehow useful.
here is folder dataset and it containes .... metadata folder and here is processing_metadata.json, for now I don't know which file produced it,
but I know it something regarding one step of clearing when from output.json->?, and here I count how many enteties I have before and how many lost (just overall, no concrete enteties, so it very small)
then I have full_dataset_v2.json, full_dataset_v3.json, full_dataset.json, tags_dataset.json, well they all produced by one file I rememeber....
Yeh, I found it is v3_data_preparation.py, it takes output.json and produce for example full_dataset_v3.json and meanwhile it also produced that processing_metadata.json
It different from version two that based on information from previous iteration, I create mapping for languages (means if for example en, eng then it all become eng) and 
I had intention to add more logging in particular to see what fall out, but for now it just how much overall lost, not how much lost due to the individual step.
the file working like this: it takes output.json. then for example I want create dataset with title, author. Then inside script I search for place where dataset forming.
Luckily I create so it possible to create array of what to extract and it will do without manually for each field. output.json have data in some json for archive from library,
so I need to go inside key and value fields of single entity and create new_key: value entity, also in my script I have mapping what to search in output.json and how it name...
currently in full_dataset_v3.json
Currently according to  v3_data_preparation.py the structure instructed of full_dataset_v3.json is: 
'full_dataset_v3.json': {"dc.title", "dc.contributor.tiedekunta", "dc.description.abstract", "dc.subject.other", "dc.subject.yso", "dc.language.iso", "dc.date.issued", "dc.identifier.uri"},
and according to processing_metadata.json:
 "total_entries": 13499,
    "deprecated_entries": 6610,
    "language_distribution": {}
I don't understand what is language_distribution for now.... I need to further inspect v3_data_preparation.py ... probably I wanted to get language of abstract marked from archive format, but
I need to do new logic to go after lang field for abstract, while before I only touched key and value... Ah, I recall, probably I have done this in another step, not sure....
well next is field_feature_extractor.py:
input_path = 'data_preparation/dataset/full_dataset.json'
    output_path = 'data_preparation/tempFieldMod/full_dataset_e.json'
.... well it do tf-idf with 500 up to features,for title, abstract and put them to corresponding product... meanwhile it suitable to pass as argument in/out, on fields which want to work with and limit of features
so this file is suitable to use by other scripts... although, I load fine for stopwords in english, but for finnish, just made manual short list of 6 stopwords.
It is likely that script not used anywhere but I need to check it.
Ah, I forgot to mention we still all under data_preparation, as we moved from data_collection.
Next is field_processor.py:
 input_file = 'data_preparation/tempFieldMod/full_dataset_er.json'
    output_file = 'data_preparation/tempFieldMod/full_dataset_erp.json'
I see it do collect tags... combine, one-hot encoding, normolization and bla, bla... but again, probably I don't use it.
Luckily I hope I could inspect if package used, is by checking imports.... for now I don't see mine imports, only libraries.
Next file is field_remover.py and here all its arguments:
input_path = 'data_preparation/tempFieldMod/full_dataset_e.json'
output_path = 'data_preparation/tempFieldMod/full_dataset_er.json'
fields_to_remove = ['thesis_title', 'abstract'].
Well, after I do that encoding or whatever, I no longer need original field... that what I thought...
hmm...maybe I could ask ai, to make me diagram showing visualization of connection between scripts, products
I see next is spllit_abstract_v2.py, this file was to split enteties. in some cases it is possible, that in one entity two abstracts...., well here is in/outs:
   input_dir = 'data_split_v3'  # Update this to match your current folder structure
    output_dir = 'data_split_v4'  # New output directory for split files
    # Files to process
    files_to_process = [
        "full_dataset_v3_train.json",
        "full_dataset_v3_val.json",
        "full_dataset_v3_test.json"
    ]
Well here is next file, it splits based on faculties, I suppose here is sufficient info about name and where:
input_path = 'data_preparation/dataset/full_dataset_v3.json'
stratify_and_split(input_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
output_folder = 'data_split_v3'
 # Define output paths
    output_train_path = os.path.join(output_folder, f'{base_name}_train.json')
    output_val_path = os.path.join(output_folder, f'{base_name}_val.json')
    output_test_path = os.path.join(output_folder, f'{base_name}_test.json')
well, then what I have done v2_data_preparation.py which made full dataset and tags dataset,
then v3_data_preparation.py which already mentioned.
Then goes visualtization_distro.py and visualization_lang.py
input_file_path = 'data_preparation/dataset/full_dataset.json'
plt.savefig('faculty_distribution.png')
and
  metadata_path = 'data_preparation/dataset/metadata/processing_metadata.json'
    visualize_language_distribution(metadata_path)
plt.savefig('language_distribution.png')
finally I finished folder data_preparation
then we go to data_split folder here is products: full_dataset and then endings test, train, val with extension json and file
split_abstract.py which ... well, my poor managment, splits into separate entities the one with several abstracts, oh, but it not adds language info:
# File paths from the existing split
files_to_process = [
    "data_split/full_dataset_train.json",
    "data_split/full_dataset_val.json",
    "data_split/full_dataset_test.json"
]

# Process each file and write the outputs to the `data_split_v2` folder
for file_path in files_to_process:
    base_name = os.path.basename(file_path)  # Extract the file name
    output_path = f"data_split_v2/{base_name}"  # Write to data_split_v2 with the same name
    split_by_abstract(file_path, output_path)

    
next folder data_split_init, looks pretty simular to previous one, except here is script stratify_split.py ... well likely I was not sure with how to do:
# Call the function with your file path and desired split ratio
input_path = 'data_preparation/dataset/full_dataset_v3.json'
stratify_and_split(input_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
 output_folder = 'data_split_v3'
 and the products output_folder_ name plus test, train, val all json.

 The next folder data_split_v3, only products here.
 The next folder data_split_v3 with only one file full_dataset_v3_train.json, don't know why maybe will se later.
 The next folder data_split_v4, this one has files with same names as in data_split_v3 folder, likely some script done transformation and as
 my style is not to modify existing files, but only to create new ones...

 next big folder data_validation.
 Faculty_encoding.py:
 input_path = 'prepared_datasets/faculty_related_v2.json'
output_pairs_path = 'prepared_datasets/unique_faculty_pairs.json'
output_singles_path = 'prepared_datasets/single_language_faculties.json'
json_data_format_analyzer.py, I see this file worked on incorrect level, instead of getting values of key, value fields, it done on the same level, so there a lot of archive format fields, with some additional marks from script:
input_file_path = 'data_collection/output.json'
output_file_path = 'data_validation/output_DataFormats.json'

json_data_integrity_checker.py, well that more correct in what to check compared to previous one, but not still:
input_file_path = 'data_collection/output.json'
    output_file_path = 'data_validation/output_FieldsStructure.json'

json_dataset_field_metric.py, this one gets useful information regarding statistics of each field aka min, max, average, same for amounts:
DEFAULT_INPUT_PATH = Path('data_preparation/dataset/full_dataset.json')
DEFAULT_OUTPUT_PATH = Path('data_validation/output_json_dataset_field_metrics.json')
well I see I have real ones:
output_json_dataset_field_metrics.json which is from intial version and long list of alerts with ids where 2 abstract, 1, 4 faculties and something else
output_json_dataset_field_metrics_5.json this one is less polluted, nicer to see

next one is json_value_occurrence_counter.py:
input_file_path = 'data_collection/output.json'
    output_file_path = 'data_validation/output_ValuesSortedByField.json'
Nice but big file, counting each value occurence under field. so I could see all unique values.

and we ended validation folder.
next one depricated_versions simular to trash bin, but I will not look here.

I suppose that one I should remove field_featureExtraction and then Title_feature_extractor.py:
input_path = 'prepared_datasets/full_details.json'
output_path = 'prepared_datasets/full_details_tfidf.json'

then folder llama_model with models downloaded from hugging face, currently I use llama_model/Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile.
then some logs folder.
then model_embdedding folder... strange name, it contains training result of models with checkpoints.
btw I have two model, the one based xlm_roberta and another one that llama, I am planning to compare their performance against each other.
---
well continue.
I see I created newer folder, here is a few results. model_outputs and it contains files aka tag_generation_results_YYYYMMDD_HHMMSS.json
Then next folder is project_utils. there is script finalizer.py ... I should run it when finnish, it probably auto commit to github, record somewhere changed files
and ask to update requirements-frozen.txt
here also project_structure.txt it contains tree of project... which script generated it? I don't know
I see next script is recordProjectStructure.py. well I suppose it know that it likely generated project_structure.txt.
and here is also requirements-frozen.txt... it created with in console command freeze. Well I hope I haven't done crusual spelling mistakes or logical due to there is 
chance what I write is not what I think, sometimes.
Then next is folder results.... Probably some model outputed its output here, because it consist of checkpoints.
then here is wandb folder. each run has saved here.
Next folder I just immideately delete it.
Next is file on root level .gitignore.
changelog.txt
command_history.txt, sometimes I forgot what I have done, so in case terminal history is cleared...
next is llama_install.py, related to llama which hermes gguf, I don't use such model now.
llama_test_a2.py related to hermes.
llama_test_a3.py related to hermes.
llama_test_a4.py related to hermes.
then here is llama-prompt.py. it seems more relevant because it addresses to model which executed on localhost8080 and probably output to console, which is likelye mozilla llamafile.
llama-test.py related to hermes.
next one is llm_generate_tags_sv2.py which uses idea how to address to llama model via program, simular to llama-prompt.py and it record results in:
'data_split/full_dataset_test.json'
well it says there is one entity to proceed, what a starnge name for sv2. why it called so and why it output its result to such place?
I am not sure if llm_generate_tags_sv3.py was working before, but I added feature... based on parameters to other from default modes, to define filepath and how much proceed,
but I don't see clear where it outputs:
output_filename = os.path.join(
                self.output_dir, 
                f'tag_generation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_filename}")

well and here input for previous script: default_dataset_path='data_split/full_dataset_test.json.
next file is llm_generate_tags.py looks simular to both previous ones, but its weird why it output in csv format, I can't open it in vs code...:
in init: output_dir='model_outputs',
                 log_dir='logs
output_df = tag_generator.process_dataset(
        'data_split/full_dataset_test.json', 
        output_path='model_outputs/generated_tags.csv'
    )
next file is llm_test_responsibility.py, it looks like this file output to console type of error code which it recieves from localhost and measure delay time,
but ... it has specific ip/8080 instead of localhost... probably I used it during debug...
main.log I don't know who generate this.
mambaforge is something related to env.
model_training_default_test.py this file is for test, which uses imdb db (not mine) and output to ./results
model_training_GCV_2.py, why I give so weird names, what is GCV? looks like I used it not long ago, because I remember issue that I not saved map of tags and here it is
probably there is chance that at some point I migrate that one-hot encoding, trancation and etc. to model training part, because teacher said that dataset should be suitable 
for many models, meanwhile model itslef should adapt clear db to its needed format.
here what it uses:  train_path: str = "data_split_v4/full_dataset_v3_train.json",
                 val_path: str = "data_split_v4/full_dataset_v3_val.json",
                 test_path: str = "data_split_v4/full_dataset_v3_test.json",
                 model_name: str = "xlm-roberta-base",
Well I hope its the correct one, because I remember now, that in some file I have defined once, but then redifined in another part...
aha, and here is where it outputs:
# Output configuration
        self.output_dir = f"model_embedding/model_output_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tokenized_data_path = "model_embedding/tokenized_datasets"
well next is model_training_GCV_singleSoft.py, probably it related to teacher suggestion about squishing input in new format-single, so I can use more simpler and robust some model training parameter
not remember which. but inputs looks simular to previous one:
train_path: str = "data_split/full_dataset_train.json",
                 val_path: str = "data_split/full_dataset_val.json",
                 test_path: str = "data_split/full_dataset_test.json",
                 model_name: str = "xlm-roberta-base",
yeah, and output also: 
# Output configuration
        self.output_dir = f"model_embedding/model_output_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tokenized_data_path = "model_embedding/tokenized_datasets"
next one is model_training_GCV.py. looks like it was my first attempt (not really, but the one which in project as trash, seems to me). I see I migrated the code from google cloud
because there I have problems with transformers library and using gpu, and btw I gain access to pretty good remote machine, well here is I/O of it:
 self.train_path = "data_split/full_dataset_train.json"
        self.val_path = "data_split/full_dataset_val.json"
        self.test_path = "data_split/full_dataset_test.json"
        # self.output_dir = f"model_embedding/model_output_{uuid.uuid4().hex[:8]}"
        self.output_dir = f"model_embedding/model_output_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tokenized_data_path = "model_embedding/tokenized_datasets"
then project_journal.txt, maybe finalizer script touched it and it was designed that later I should type here?
then README.md. Well, probaly I haven't update it long.... but there was attempt to explain what is project files and its structure....
requirements.txt it looks like it was initial when I done with frozen, but I don't like location, and I see it has long name for requirements as if address.
tags.pickle, save file from one of the model_trainging_GCV_2.py, I think.
temp_notes.txt, well some notes regarding remarks.
then here is testing_output_2.py:
# Load test dataset
test_data = Dataset.from_json("data_split_v4/full_dataset_v3_test.json")

# Load model, tokenizer, and tag2id mapping
model_name = "xlm-roberta-base"  # Replace with your trained model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("model_embedding/model_output_20241213_141245/checkpoint-668")
and then here is testing_output_3.py:
test_data = Dataset.from_json("data_split/full_dataset_test.json")

# Load model, tokenizer, and tag2id mapping
model_name = "xlm-roberta-base"  # Replace with your trained model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("model_embedding/model_output_20241128_110456/checkpoint-130")
"""
"""
well probably month passed, today is 28.02.2025
again let's think how properly organise...
pipeline.... Let's not touch louding dataset
Firstly I need to get correct field, then check for mistakes, then again filter....
Then I choose model and further prepare dataset under this model for example one hot encoding, augmentation with additional field defining fomrattng)....
then I start training, I get checkpoints and pickles
then I test output.... and then I got it, but what if it bad? do I continue or change parameters.....?
"""
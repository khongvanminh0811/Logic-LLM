import itertools
import string
import json
import os
import time
import re
from tqdm import tqdm
from utils import OpenAIModel
import argparse
import openai
import random
from datasets import load_dataset

class NL2FOLGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path

        self.openai_api = OpenAIModel(args.model_name, args.stop_words, args.max_new_tokens)
        self.rules_facts_prompt_creator = {'NarativeQA': self.rules_facts_prompts_narrativeqa,
                                            'HotpotQA': self.rules_facts_prompt_hotpotqa}
        self.entities_prompt_creator = {'NarativeQA': self.entities_prompt_narrativeqa,
                                        'HotpotQA': self.entities_prompt_hotpotqa}
        # self.predicates_prompt_creator = {'NarativeQA': self.predicates_prompt_narrativeqa}
        self.fol_extract_prompt_creator = {'NarativeQA': self.fol_extract_prompt_narrativeqa,
                                            'HotpotQA': self.fol_extract_prompt_hotpotqa}
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/custom'
        rules_facts_extract_prompt_file = f'{prompt_file}/rules-facts-extract.txt'
        entities_prompt_file = f'{prompt_file}/entities.txt'
        predicates_prompt_file = f'{prompt_file}/predicates.txt'
        fol_extract_prompt_file = f'{prompt_file}/fol-extract.txt'
        with open(rules_facts_extract_prompt_file, 'r') as f:
            self.rules_facts_extract_prompt = f.read()
        with open(entities_prompt_file, 'r') as f:
            self.entities_prompt = f.read()
        with open(predicates_prompt_file, 'r') as f:
            self.predicates_prompt = f.read()
        with open(fol_extract_prompt_file, 'r') as f:
            self.fol_extract_prompt = f.read()
    
    def rules_facts_prompts_narrativeqa(self, test_data):
        context = test_data['summary'].split('\n')
        rules_facts_prompts = [self.rules_facts_extract_prompt.replace('[[CONTEXT]]', paragraph) for paragraph in context]
        return rules_facts_prompts
    
    def rules_facts_prompt_hotpotqa(self, test_data):
        context = test_data['context'].split('\n')
        rules_facts_prompts = [self.rules_facts_extract_prompt.replace('[[CONTEXT]]', paragraph) for paragraph in context]
        return rules_facts_prompts
    
    def entities_prompt_narrativeqa(self, test_data, facts_rules):
        context = test_data['summary'].split('\n')
        entities_prompts = []
        for i, paragraph in enumerate(context):
            facts = facts_rules[i]['facts']
            rules = facts_rules[i]['rules']

            entities_prompt = self.entities_prompt.replace('[[CONTEXT]]', paragraph).replace('[[FACTS]]', '\n'.join(facts)).replace('[[RULES]]', '\n'.join(rules))
            entities_prompts.append(entities_prompt)
        return entities_prompts

    def entities_prompt_hotpotqa(self, test_data, facts_rules):
        context = test_data['context'].split('\n')
        entities_prompts = []
        for i, paragraph in enumerate(context):
            facts = facts_rules[i]['facts']
            rules = facts_rules[i]['rules']

            entities_prompt = self.entities_prompt.replace('[[CONTEXT]]', paragraph).replace('[[FACTS]]', '\n'.join(facts)).replace('[[RULES]]', '\n'.join(rules))
            entities_prompts.append(entities_prompt)
        return entities_prompts
    # def predicates_prompt_narrativeqa(self, sentences, entities):
    #     predicates_prompts = [self.predicates_prompt.replace('[[SENTENCE]]', sentence).replace('[[ENTITIES]]', ', '.join(entities)) for sentence in sentences]
    #     return predicates_prompts
    def fol_extract_prompt_narrativeqa(self, test_data, facts_rules, entities):
        context = test_data['summary'].split('\n')
        fol_extract_prompts = []
        for i, paragraph in enumerate(context):
            rules = facts_rules[i]['rules']

            fol_extract_prompt = self.fol_extract_prompt.replace('[[CONTEXT]]', paragraph).replace('[[RULES]]', '\n'.join(rules)).replace('[[ENTITIES]]', ', '.join(entities))
            fol_extract_prompts.append(fol_extract_prompt)
        return fol_extract_prompts

    def fol_extract_prompt_hotpotqa(self, test_data, facts_rules, entities):
        context = test_data['context'].split('\n')
        fol_extract_prompts = []
        for i, paragraph in enumerate(context):
            rules = facts_rules[i]['rules']

            fol_extract_prompt = self.fol_extract_prompt.replace('[[CONTEXT]]', paragraph).replace('[[RULES]]', '\n'.join(rules)).replace('[[ENTITIES]]', ', '.join(entities))
            fol_extract_prompts.append(fol_extract_prompt)
        return fol_extract_prompts

    def load_raw_dataset(self, split):
        print(f"Loading {os.path.join(self.data_path, self.dataset_name, f'{split}.json')}...")
        if self.dataset_name == 'NarativeQA':
            with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json'),  encoding="utf-8") as f:
                raw_dataset = json.load(f)
            # dataset = load_dataset("deepmind/narrativeqa")[split]
            # all_ids = list(set(sample['document']['id'] for sample in dataset))
            # sample_ids = random.sample(all_ids, 50)  # Sample 50 unique IDs
            # filtered_dataset = dataset.filter(lambda x: x['document']['id'] in sample_ids)
            
            # Transform into desired structure
            processed_dataset = []
            documents = {}
            for idx, example in enumerate(raw_dataset):
                context = example['context']
                summary = example['summary']
                question = example['question'].strip()
                answer = example['answer'].strip()
                context_id = example['context_id']

                processed_dataset.append({
                    "id": idx,
                    "context": context,
                    "summary": summary,
                    "question": question,
                    "answer": answer,
                    "context_id": context_id
                })
                if context_id not in documents:
                    documents[context_id] = {
                        "id": context_id,
                        "context": context,
                        "summary": summary
                    }
            self.documents = list(documents.values())
            # with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json'), 'w', encoding="utf-8") as f:
            #     json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
            print(f"Loaded {len(processed_dataset)} examples from {split} split.")
            return processed_dataset
        if self.dataset_name == 'HotpotQA':
            with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
                raw_dataset = json.load(f)
            
            filtered_dataset = random.sample(raw_dataset, 200)
            processed_dataset = []
            documents = {}
            for idx, sample in enumerate(filtered_dataset):
                context = '\n'.join([' '.join(info[1]) for info in raw_dataset['context']])
                processed_dataset.append({
                    "id": idx,
                    "context": context,
                    "question": example['question'],
                    "answer": example['answer'],
                    "context_id": sample['_id']
                })
                documents[sample['_id']] = {
                    "id": sample['_id'],
                    "context": context,
                }
            self.documents = list(documents.values())
            return processed_dataset
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def nl_to_fol(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        documents = self.documents
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_documents.json'), 'w', encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        facts_rules_outputs = {}
        entities_outputs = {}
        fol_outputs = {}

        for i in tqdm(range(len(documents))):
            rules_facts_prompts = self.rules_facts_prompt_creator[self.dataset_name](documents[i])
            facts_rules_responses = self.openai_api.batch_generate(rules_facts_prompts)
            exist = False
            for j, response in enumerate(facts_rules_responses):
                facts, rules = extract_facts_and_rules(response)
                if exist:
                    facts_rules_outputs[documents[i]['id']].append({
                        'paragraph_id': j,
                        'facts': facts,
                        'rules': rules
                    })
                else:
                    facts_rules_outputs[documents[i]['id']] = [{
                        'paragraph_id': j,
                        'facts': facts,
                        'rules': rules
                    }]
                    exist = True
            
            entities_prompts = self.entities_prompt_creator[self.dataset_name](documents[i], facts_rules_outputs[documents[i]['id']])
            entities_responses = self.openai_api.batch_generate(entities_prompts)
            exist = False
            for j, response in enumerate(entities_responses):
                entities = response.split('\n')
                if exist:
                    entities_outputs[documents[i]['id']].append({
                        'id': documents[i]['id'],
                        'paragraph_id': j,
                        'entities': entities
                    })
                else:
                    entities_outputs[documents[i]['id']] = [{
                        'id': documents[i]['id'],
                        'paragraph_id': j,
                        'entities': entities
                    }]
                    exist = True
            
            entities_list = []
            group_entities = {}
            exist = False
            for line in entities:
                groups = line.split(':')
                if len(groups) == 1:
                    entities_split = [entity.strip() for entity in line.split(',')]
                    entities_list.extend(entities_split)
                    if exist:
                        group_entities[""].extend(entities_split)
                    else:
                        group_entities[""] = entities_split
                    exist = True
                else:
                    group_name = groups[0].strip()
                    group_entities[group_name] = [entity.strip() for entity in groups[1].split(',')]
                    entities_list.extend(group_name)
                    entities_list.extend(group_entities[group_name])
        
            chars = generate_letter()
            marked_entities = []
            for group_name in group_entities:
                if group_name != "":
                    current_char = next(chars)
                    marked_entities.append(f'{group_name}: {current_char}_0')
                    marked_entities.extend([f'{group_entity}: {current_char}_{i+1}' for (i, group_entity) in enumerate(group_entities[group_name])])
            if exist:
                marked_entities.extend([f'{group_entity}: {next(chars)}' for group_entity in group_entities['']])        

            # paragraphs = documents[i]['summary'].split('\n')
            # predicates_outputs = []
            # for j, _ in enumerate(paragraphs):
            #     facts_rules = facts_rules_outputs[j]
            #     fact_predicates_extract_prompt = self.predicates_prompt_creator[self.dataset_name](facts_rules["facts"], marked_entities)
            #     rule_predicates_extract_prompt = self.predicates_prompt_creator[self.dataset_name](facts_rules["rules"], marked_entities)
            #     fact_predicates_response = self.openai_api.batch_generate(fact_predicates_extract_prompt)
            #     time.sleep(1)
            #     rule_predicates_response = self.openai_api.batch_generate(rule_predicates_extract_prompt)
            #     time.sleep(1)
            #     predicates_outputs.append({
            #         'id': documents[i]['id'],
            #         'paragraph_id': j,
            #         'facts_predicates': fact_predicates_response,
            #         'rules_predicates': rule_predicates_response,
            #         'entities': marked_entities
            #     })
            # with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_predicates_extract.json'), 'w', encoding="utf-8") as f:
            #     json.dump(predicates_outputs, f, indent=2, ensure_ascii=False)
            
            fol_extract_prompts = self.fol_extract_prompt_creator[self.dataset_name](documents[i], facts_rules_outputs[documents[i]['id']], marked_entities)
            fol_responses = self.openai_api.batch_generate(fol_extract_prompts)
            time.sleep(1)
            exist = False
            for j, response in enumerate(fol_responses):
                fols = [fix_misencoded_logical_symbols(fol) for fol in response.strip().split('\n')]
                if exist:
                    fol_outputs[documents[i]['id']].append({
                        'id': documents[i]['id'],
                        'paragraph_id': j,
                        'fols': fols
                    })
                else:
                    fol_outputs[documents[i]['id']] = [{
                        'id': documents[i]['id'],
                        'paragraph_id': j,
                        'fols': fols
                    }]
                    exist = True
                print(f"Paragraph {j} FOLs: {[fix_misencoded_logical_symbols(fol) for fol in fols]}")

            # coref
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_rules_facts_extract.json'), 'w', encoding="utf-8") as f:
            json.dump(facts_rules_outputs, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_entities_extract.json'), 'w', encoding="utf-8") as f:
            json.dump(entities_outputs, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_fol_extract.json'), 'w', encoding="utf-8") as f:
            json.dump(fol_outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=random.randint(0, 1000))
    args = parser.parse_args()
    return args

def fix_misencoded_logical_symbols(text):
    replacements = {
        'âˆ§': '∧',   # AND
        'âˆ¨': '∨',   # OR
        'âŠ•': '⊕',   # XOR
        'Â¬':  '¬',   # NOT
        'â†’': '→',   # IMPLIES
        'â†”': '↔',   # IFF
        'âˆ€': '∀',   # FORALL
        'âˆƒ': '∃',   # EXISTS
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

def generate_letter():
    letters = string.ascii_lowercase  # 'a' to 'z'
    length = 1
    while True:
        for combo in itertools.product(letters, repeat=length):
            yield ''.join(combo)
        length += 1

def extract_facts_and_rules(text):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    facts = []
    rules = []
    section = None

    for line in lines:
        if line.startswith("Facts:"):
            section = "facts"
            continue
        elif line.startswith("Rules:"):
            section = "rules"
            continue

        if section == "facts":
            facts.append(line)
        elif section == "rules":
            rules.append(line)

    return facts, rules
    
if __name__ == '__main__':
    # with open(os.path.join('.\data', 'LogiQA', 'test.json')) as f:
    #     raw_dataset = json.load(f)
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.api_key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    random.seed(args.seed)

    nl2fol_generator = NL2FOLGenerator(args)
    nl2fol_generator.nl_to_fol()

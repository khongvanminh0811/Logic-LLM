import json
import os
import re
import time
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
# from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
# from symbolic_solvers.csp_solver.csp_solver import CSP_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
from utils import OpenAIModel
import argparse
import openai
import random
from datasets import load_dataset

class CandidateAnswersGenerator:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_path = args.save_path

        self.openai_api = OpenAIModel(args.model_name, args.stop_words, args.max_new_tokens)

        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'NarativeQA': FOL_Prover9_Program,
                                # 'ProntoQA': Pyke_Program, 
                                # 'ProofWriter': Pyke_Program,
                                # 'LogicalDeduction': CSP_Program,
                                'AR-LSAT': LSAT_Z3_Program}
        self.program_executor = program_executor_map[self.dataset_name]

        self.candidate_answers_prompt_creator = {"NarativeQA": self.candidate_answers_prompt_narativeqa}
        self.question_template_prompt_creator = {"NarativeQA": self.question_template_prompt_narativeqa}
        self.candidate_answers_to_fol_prompt_creator = { "NarativeQA": self.candidate_answers_to_fol_prompt_narativeqa }
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/custom'
        candidate_answers_prompt_file = f'{prompt_file}/CoT.txt'
        question_template_prompt_file = f'{prompt_file}/question-templates.txt'
        candidate_answer_fol_prompt_file = f'{prompt_file}/candidate-answers-fol.txt'
        with open(candidate_answers_prompt_file, 'r', encoding='utf-8') as f:
            self.candidate_answers_prompt = f.read()
        with open(question_template_prompt_file, 'r', encoding='utf-8') as f:
            self.question_template_prompt = f.read()
        with open(candidate_answer_fol_prompt_file, 'r', encoding='utf-8') as f:
            self.candidate_answers_to_fol_prompt = f.read()

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
            with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json'), 'w', encoding="utf-8") as f:
                json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
            return processed_dataset
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def candidate_answers_prompt_narativeqa(self, test_data, paragraph):
        question = test_data['question'].strip()
        full_prompt = self.candidate_answers_prompt.replace('[[CONTEXT]]', paragraph).replace('[[QUESTION]]', question)
        return full_prompt
    
    def question_template_prompt_narativeqa(self, question, candidate_answers):
        candidate_answers_str = '\n'.join(candidate_answers)
        full_prompt = self.question_template_prompt.replace('[[QUESTION]]', question).replace('[[CANDIDATE_ANSWERS]]', candidate_answers_str)
        return full_prompt

    def candidate_answers_to_fol_prompt_narativeqa(self, paragraph, predicates, premises, template, candidate_answers):
        prompts = []
        for candidate_answer in candidate_answers:
            candidate_answer_template = template.replace('[[CANDIDATE_ANSWER]]', candidate_answer)  
            prompt = self.candidate_answers_to_fol_prompt.replace('[[PARAGRAPH]]', paragraph).replace('[[PREDICATES]]', '\n'.join(predicates)).replace('[[PREMISES]]', '\n'.join(premises)).replace('[[CONCLUSION_SENTENCE]]', candidate_answer_template)
            prompts.append(prompt)
        return prompts
        
    def generate_candidate_answers(self):
        raw_dataset = self.load_raw_dataset(self.split)
        
        outputs = {}
        chunk_size = 10
        for i, test_data in tqdm(enumerate(raw_dataset)):
            paragraphs = raw_dataset[i]['summary'].split('\n')
            candidate_answer_prompts = [self.candidate_answers_prompt_creator[self.dataset_name](test_data, paragraph) for paragraph in paragraphs]
            
            if (9 * (len(raw_dataset) / chunk_size) <= i) :
                while True:
                    try:
                        results = self.openai_api.batch_generate(candidate_answer_prompts)
                        break
                    except openai.RateLimitError as e:
                        print(f"Rate limit exceeded, retrying after 2 seconds: {e}")
                        time.sleep(2)
                exist = False

                for j in range(len(paragraphs)):
                    lines = results[j].strip().splitlines()
                    candidate_answers = []
                    reasoning = ""
                    correct_answer = ""
                    mode = None

                    for line in lines:
                        line = line.strip()
                        if line == "Candidate Answers:":
                            mode = "candidate"
                            continue
                        elif line == "Correct Answer:":
                            mode = "correct"
                            continue
                        elif line == "Reasoning:":
                            mode = "reasoning"
                            continue

                        if mode == "candidate" and line:
                            line = re.sub(r'\([^)]*\)', '', line)  # Remove parentheses and their content
                            line = re.sub(r'\s+', ' ', line).strip()
                            if line != '':
                                candidate_answers.append(line)
                        elif mode == "correct" and line:
                            line = re.sub(r'\([^)]*\)', '', line)  # Remove parentheses and their content
                            line = re.sub(r'\s+', ' ', line).strip()
                            correct_answer = line
                        elif mode == "reasoning" and line:
                            reasoning += line + " "
                    
                    if not exist: 
                        outputs[test_data['id']] = [{
                            'id': f"{test_data['id']}",
                            'paragraph_id': j,
                            'context_id': test_data['context_id'],
                            'paragraph': paragraphs[j],
                            'context': test_data['summary'],
                            'question': test_data['question'],
                            'candidate_answers': candidate_answers,
                            'reasoning': reasoning,
                            'correct_answer': correct_answer,
                        }]
                        exist = True
                        
                    else:
                        outputs[test_data['id']].append({
                            'id': f"{test_data['id']}",
                            'paragraph_id': j,
                            'paragraph': paragraphs[j],
                            'context': test_data['summary'],
                            'context_id': test_data['context_id'],
                            'question': test_data['question'],
                            'candidate_answers': candidate_answers,
                            'reasoning': reasoning,
                            'correct_answer': correct_answer,
                        })
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_candidate_answers_10.json'), 'w', encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def generate_question_template_and_answer_fol(self):
        raw_dataset = self.load_raw_dataset(self.split)
        candidate_answers_outputs_id_dict = {}
        candidate_answers_outputs_context_dict = {}
        candidate_answers_outputs = []
        for i in range(10):
            with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_candidate_answers_{i+1}.json'), 'r', encoding="utf-8") as f:
                data= json.load(f)
                candidate_answers_outputs_id_dict = {**candidate_answers_outputs_id_dict, **data}
                
                for value in list(data.values()):
                    candidate_answers_outputs.extend(value)

        for i, candidate_answers_output in enumerate(candidate_answers_outputs):
            key = candidate_answers_output['context_id']
            if key not in candidate_answers_outputs_context_dict:
                candidate_answers_outputs_context_dict[key] = [candidate_answers_output]
            else:
                candidate_answers_outputs_context_dict[key].append(candidate_answers_output)
        
        # question_templates_prompts = []
        # for i, test_data in enumerate(raw_dataset):
        #     candidate_answers_output = candidate_answers_outputs_context_dict[test_data['context_id']]
        #     candidate_answers = []
        #     for output in candidate_answers_output:
        #         candidate_answers.extend(output['candidate_answers'])
            
        #     question = test_data['question'].strip()
        #     question_template_prompt = self.question_template_prompt_creator[self.dataset_name](question, candidate_answers)
            
        #     question_templates_prompts.append(question_template_prompt)
        
        # template_results = []
        # for i in tqdm(range(15*int(len(question_templates_prompts)/20) - 10, len(question_templates_prompts), 10)):
        #     while True:
        #         try:
        #             results = self.openai_api.batch_generate(question_templates_prompts[i:i+10])
        #             template_results.extend(results)
        #             break
        #         except openai.RateLimitError as e:
        #             print(f"Rate limit exceeded, retrying after 2 seconds: {e}")
        #             time.sleep(10)

        
        # question_templates = []
        # question_templates_outputs = []
        # for i, result in enumerate(template_results):  
        #     question_templates.append(result.strip())
        #     question_templates_outputs.append({
        #         'id': raw_dataset[i]['id'],
        #         'question': raw_dataset[i]['question'],
        #         'context_id': raw_dataset[i]['context_id'],
        #         'question_template': result.strip(),
        #     })
           
        # with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_question_templates_7.json'), 'w', encoding="utf-8") as f:
        #     json.dump(question_templates_outputs, f, indent=2, ensure_ascii=False)
        
        question_templates_outputs = []
        for i in range(7):
            with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_question_templates_{i+1}.json'), 'r', encoding="utf-8") as f:
                data= json.load(f)
                question_templates_outputs.extend(data)

        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_gpt-4o-mini_fol_extract.json'), 'r', encoding="utf-8") as f:
            fols_output = json.load(f)

        # with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_hypothesis_1.json'), 'w', encoding="utf-8") as f:
        #     dump_outputs = json.load(f)

        dump_outputs = {}
        chunk_size = 5
        for i, test_data in tqdm(enumerate(raw_dataset)):
            if (i >= 4 * (len(raw_dataset) / chunk_size)) :
                question_template = question_templates_outputs[i]['question_template']
                context_id = test_data['context_id']
                paragraphs = raw_dataset[i]['summary'].split('\n')
                fols = fols_output[context_id]
                candidate_answers_output =  candidate_answers_outputs_id_dict[str(i)]
                for j, fol in enumerate(fols):
                    lines = fol['fols']
                    premises = []
                    predicates = []
                    mode = None
                    for line in lines:
                        line = line.strip()
                        if line == "Premises:":
                            mode = "premises"
                            continue
                        elif line == "Predicates:":
                            mode = "predicates"
                            continue

                        if mode == "premises" and line:
                            if line.strip() != '':
                                premises.append(line)
                        elif mode == "predicates" and line:
                            if line.strip() != '':
                                predicates.append(line)

                    candidate_answers = []
                    for candidate_answer in candidate_answers_output[j]['candidate_answers']:
                        candidate_answers.append(candidate_answer)

                    if len(candidate_answers) == 0:
                        print(f"i = {i}, No candidate answers found for context_id {context_id}, paragraph {j}. Skipping...")
                        continue

                    fols_prompts = self.candidate_answers_to_fol_prompt_creator[self.dataset_name](paragraphs[j], predicates, premises, question_template, candidate_answers)

                    while True:
                        try:
                            results = self.openai_api.batch_generate(fols_prompts)
                            break
                        except openai.RateLimitError as e:
                            print(f"Rate limit exceeded, retrying after 2 seconds: {e}")
                            time.sleep(10)
                    
                    if i not in dump_outputs:
                        dump_outputs[i] = [{
                            'id': i,
                            'context_id': context_id,
                            'paragraph_id': j,
                            'paragraph': paragraphs[j],
                            'question_template': question_template,
                            'candidate_answers': candidate_answers,
                            'predicates': predicates,
                            'premises': premises,
                            'conclusions': results
                        }]
                    else:
                        dump_outputs[i].append({
                            'id': test_data['id'],
                            'context_id': context_id,
                            'paragraph_id': j,
                            'paragraph': paragraphs[j],
                            'question_template': question_template,
                            'candidate_answers': candidate_answers,
                            'predicates': predicates,
                            'premises': premises,
                            'conclusions': results
                        })
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_hypothesis_5.json'), 'w', encoding="utf-8") as f:
            json.dump(dump_outputs, f, indent=2, ensure_ascii=False)

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

    
if __name__ == '__main__':
    # with open(os.path.join('.\data', 'LogiQA', 'test.json')) as f:
    #     raw_dataset = json.load(f)
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.api_key
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    random.seed(args.seed)
    
    nl2fol_generator = CandidateAnswersGenerator(args)
    nl2fol_generator.generate_candidate_answers()
    # nl2fol_generator.generate_question_template_and_answer_fol()

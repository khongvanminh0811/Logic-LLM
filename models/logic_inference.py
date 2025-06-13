import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
# from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
# from symbolic_solvers.csp_solver.csp_solver import CSP_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random
from backup_answer_generation import Backup_Answer_Generator

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.backup_strategy = args.backup_strategy

        self.dataset = self.load_logic_programs()
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'NarativeQA': FOL_Prover9_Program,
                                # 'ProntoQA': Pyke_Program, 
                                # 'ProofWriter': Pyke_Program,
                                # 'LogicalDeduction': CSP_Program,
                                'AR-LSAT': LSAT_Z3_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)

    def load_logic_programs(self):
        if self.dataset_name == 'NarativeQA':
            dataset = {}
            for i in range(5):
                with open(os.path.join('./outputs/logic_programs', f'{self.dataset_name}_{self.split}_{self.model_name}_hypothesis_{i+1}.json'), 'r', encoding='utf-8') as f:
                    dataset = {**dataset, **json.load(f)}
            return dataset
        with open(os.path.join('./outputs/logic_programs', f'{self.dataset_name}_{self.split}_{self.model_name}.json')) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_backup-{self.backup_strategy}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(logic_program, self.dataset_name)
        # cannot parse the program
        if program.flag == False:
            return '', 'parsing error', ''
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            return '', 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', ''

    def inference_on_dataset(self):
        outputs = {}
        error_count = 0
        program_count = 0
        
        for i in tqdm(self.dataset):
            if self.dataset[i][0]['id'] > 1:
                break
            for j in range(len(self.dataset[i])):
                example = self.dataset[i][j]
                flags = []
                answers = []
                for k in range(len(example['conclusions'])):
                    if example['conclusions'][k] == 'None':
                        continue
                    
                    predicates = example['predicates']
                    premises = example['premises']
                    conclusion = example['conclusions'][k]
                    raw_answer = example['candidate_answers'][k]
                    raw_logic_program = 'Predicates:\n' + '\n'.join(predicates) + '\nPremises:\n' + '\n'.join(premises) + '\nConclusion:\n' + conclusion + ' ::: ' + raw_answer
                    print('raw_logic_program: ', raw_logic_program)
                    # execute the logic program
                    answer, flag, error_message = self.safe_execute_program(i, raw_logic_program)
                    program_count += 1

                    if not flag == 'success':
                        error_count += 1
                    flags.append(flag)
                    answers.append(answer)
                # create output
                output = {
                        'id': i,
                        'paragraph': j,
                        'flags': flags,
                        'predicted_answers': answers,
                    }
                if i not in outputs:
                    outputs[i] = [output]
                else:
                    outputs[i].append(output)
        # for i in self.dataset:
        #     if i not in outputs:
        #         outputs[i] = [{
        #             'id': i,
        #             'paragraph': -1,
        #             'flags': 'no candidate answers',
        #             'predicted_answers': self.backup_generator.get_backup_answer(i)
        #         }]
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        pass
        # complied_krb_dir = './models/compiled_krb'
        # if os.path.exists(complied_krb_dir):
        #     print('removing compiled_krb')
        #     os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='./baselines/results')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # engine = LogicInferenceEngine(args)
    # engine.inference_on_dataset()
    logic_program = """Predicates:
    Saved(x, y) ::: x saves y.
    Orphan(x) ::: x is an orphan child.
    DisguisedAs(x, y) ::: x is disguised as y.
    Hides(x, y) ::: x hides y.
    Fled(x, y) ::: x has fled from y.
    Burned(x) ::: x is burned.
    Killed(x) ::: x is killed.
    Sent(x, y) ::: x is sent to y.
    BelievedDead(x) ::: x is believed to have died.
    Premises:
    Fled(kingCharlesII, London) ::: King Charles I has fled from London.
    Sent(parliamentarySoldiers, NewForest) ::: Parliamentary soldiers have been sent to search the New Forest.
    Burned(arnwoodEstate) ::: Arnwood, the house of Colonel Beverley, is burned.
    ∃x (Orphan(x) ∧ BelievedDead(x)) ::: The four orphan children are believed to have died in the flames.
    Saved(jacobArmitage, a_5) ::: Jacob Armitage saves Edward's sisters.
    Saved(jacobArmitage, a_6) ::: Jacob Armitage saves Edward's brother.
    Hides(jacobArmitage, a_0) ::: Jacob Armitage hides the orphan children in his isolated cottage.
    DisguisedAs(a_5, grandchildren) ::: Edward's sisters are disguised as Jacob Armitage's grandchildren.
    DisguisedAs(a_6, grandchildren) ::: Edward's brother is disguised as Jacob Armitage's grandchildren.
    Conclusion:
    Saved(jacobArmitage, a_0) ∧ Saved(jacobArmitage, a_5) ∧ Saved(jacobArmitage, a_6) ::: A local verderer"""
    prover9_program = FOL_Prover9_Program(logic_program)
    print(prover9_program.flag)
    answer, error_message = prover9_program.execute_program()
    print(answer)
    print(error_message)
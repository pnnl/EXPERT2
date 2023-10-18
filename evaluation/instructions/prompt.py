
class PromptGenerator:
    @staticmethod
    def get_prompt(task, task_type, TASK_CATS, result):
        prompt = {"binary_classif": f"### Below is an input containing a title-abstract pair. Classify this input into one of the possible {task} categories. ### Possible Categories: {TASK_CATS} ### Input: ## Title: {result['title']} ## Abstract: {result['abstract']} ### Response:", 
                  "multilabel_classif": f"### Below is an input containing a title-abstract pair. Classify this input into one or more possible {task} categories. ### Possible Categories: {TASK_CATS} ### Input: ## Title: {result['title']} ## Abstract: {result['abstract']} ### Response:", 
                  "singlelabel_classif": f"### Below is an input containing a title-abstract pair. Classify this input into one of the possible {task} categories. ### Possible Categories: {TASK_CATS} ### Input: ## Title: {result['title']} ## Abstract: {result['abstract']} ### Response:", 
                  "open_domain": "" 
             }

        return prompt[task_type]
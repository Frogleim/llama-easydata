import ollama
import json

all_results = []


def read_output():
    for i in range(1, 11):
        print(i)
        with open(f'./files/Court Samples OCR/{i}.txt', 'r', encoding='utf-8') as output_data:
            text = output_data.read()
            prompt = build_prompt(text)
            run_model(prompt, i)


def build_prompt(document_data):
    prompt = f'''
        You are tasked with extracting and formatting monetary values from a legal document. 
        Do not infer or assume any information outside of the document. Follow these instructions strictly:
        1. **Monetary Value**: Extract any monetary value in the format "XXXX,YY руб" from the document. 
           - Ensure that the integer part is separated by spaces (i.e., "XXXX руб").
           - Convert the fractional part after the comma to "YY коп." If there is no fractional part, use "00 коп.".
           - Ensure no extra zeroes are added, and format as "XXXX руб. YY коп.".
        2. Any field not present or incomplete should strictly return "Не указано".

        Document data begins below:

        =================
        {document_data}
        =================

        Output the result in the following JSON structure:
        {{
            "Amount": "XXXX руб. YY коп."
        }}
        Only output JSON and nothing else.
        '''
    save_all_responses()

    return prompt


def run_model(prompt, file_number):
    try:
        response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
        json_response = response  # Ensure to strip any extra spaces/newlines
        parsed_data = json_response['message']['content']
        all_results.append({
            "file_number": file_number,
            "response": parsed_data
        })
    except Exception as e:
        print(f"An error occurred while processing file {file_number}: {e}")


def save_all_responses():
    try:
        # Save all accumulated results in a single JSON file
        with open('./output/all_responses.json', 'w', encoding='utf-8') as json_file:
            json.dump(all_results, json_file, ensure_ascii=False, indent=4)
        print("All responses have been saved in one JSON file.")
    except Exception as e:
        print(f"An error occurred while saving all responses: {e}")


if __name__ == '__main__':
    read_output()

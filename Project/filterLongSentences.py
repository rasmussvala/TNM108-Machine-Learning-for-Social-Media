def filter_long_answers(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    filtered_lines = []
    question_indices = [i for i in range(len(lines)) if i % 2 == 0]

    for idx in question_indices:
        question = lines[idx].strip()
        answer = lines[idx + 1].strip()

        max_words = 10

        if len(question.split()) <= max_words and len(answer.split()) <= max_words:
            filtered_lines.append(question + "\n")
            filtered_lines.append(answer + "\n")

    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(filtered_lines)


# Usage
input_file_path = "topicalChat.txt"
output_file_path = "topicalChatFiltered.txt"
filter_long_answers(input_file_path, output_file_path)

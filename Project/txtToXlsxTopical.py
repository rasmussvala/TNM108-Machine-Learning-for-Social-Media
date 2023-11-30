import pandas as pd

# Read the text file
with open("topicalChatFiltered.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Create lists to store data for cells A, B, and C
cell_a = []
cell_b = []
cell_c = []

# Iterate through the lines to separate questions and answers
for i, line in enumerate(lines):
    line = line.strip()
    if i % 2 == 0:  # Odd rows are questions
        cell_a.append(i // 2 + 1)  # Number of the q-a pair
        cell_b.append(line)  # Question in cell B
        cell_c.append("")  # Empty answer in cell C for now
    else:  # Even rows are answers
        cell_c[-1] = line  # Assign the answer to the last question

# Create a DataFrame with the formatted data
data = {" ": cell_a, "question": cell_b, "answer": cell_c}
new_df = pd.DataFrame(data)

# Save the new DataFrame to an Excel file
new_df.to_excel("datasets/topicalChatFiltered.xlsx", index=False)

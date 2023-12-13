def letter_to_number(letter):
    return str(ord(letter.lower()) - ord('a'))


input_file_path = 'train_data/player_game_list_txt_3.txt'
output_file_path = 'train_data/player_game_list_move.txt'

finalList = []

with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if 'SZ[11]' in line:
            moveList = []
            # Split the line at 'SZ[11]' and replace the part after it
            parts = line.split(';')[2:-1]
            for i in parts:
                if i[2:-1] == "swap":
                    moveList.append(moveList[0])
                else:
                    moveList.append(i[2].capitalize()+letter_to_number(i[3]))
            finalList.append(' '.join(moveList))

with open(output_file_path, "a") as f:
    for item in finalList:
        f.write("%s\n" % item)

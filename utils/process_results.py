import os
import sys

if len(sys.argv) < 2:
    print('USAGE: ' + sys.argv[0] + ' {folder}')
    exit(0)

folder_name = sys.argv[1]

if not os.path.exists(folder_name):
    print('Folder not found')
    exit(0)

with open(folder_name + '.txt', 'w') as output:
    for result_filename in os.listdir(folder_name):
        result_filepath = os.path.join(folder_name, result_filename)
        with open(result_filepath, 'r') as result_file:
            result_content = result_file.readlines()
            output.writelines(result_content)
            output.write('\n')
import json

nb_path = 'c:/Users/pc/LabsDLA/Lab4/Lab4_OOD.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        print(f"--- Outputs for Cell {i} ---")
        for out in cell['outputs']:
            if out['output_type'] == 'stream':
                print(out['text'])
            elif out['output_type'] == 'error':
                print("ERROR:", out['ename'], out['evalue'])
            elif out['output_type'] == 'display_data' or out['output_type'] == 'execute_result':
                if 'text/plain' in out['data']:
                    print(out['data']['text/plain'])
                if 'image/png' in out['data']:
                    print("[Image Generated]")

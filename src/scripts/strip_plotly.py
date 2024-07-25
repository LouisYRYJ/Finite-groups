#!/usr/bin/env python
import nbformat

def remove_plotly_output(notebook_path, output_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Iterate through cells
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'outputs' in cell:
                # Remove Plotly outputs
                cell['outputs'] = [
                    output for output in cell['outputs']
                    if not (output.get('data', {}).get('application/vnd.plotly.v1+json')
                            or output.get('data', {}).get('text/html'))
                ]
    
    # Write the modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Usage
if __name__ == '__main__':
    import sys
    notebook_path = sys.argv[1]
    remove_plotly_output(notebook_path, notebook_path)
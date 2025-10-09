# ppi_iterative_opt
Protein protein interaction iterative optimization strategy to improve Alphafold2 validation metrics

## Description
This script takes a single PDB file of a protein-protein interaction (PPI) complex as input and will try to improve Alphafold2 structure prediction validation metrics by cycling through iterations of RFdiffusion (partial diffusion), Protein MPNN, and Alphafold2.

## Reference
.

## Installation
You can clone this repo into a preferred destination directory by going to that directory and then running:

`git clone https://github.com/davidekim/ppi_iterative_opt.git`

Open the script in a text editor and edit the configuration parameters to point to your RFDiffusion, Protein MPNN, and Alphafold2 installations. 

## Usage
ppi_iterative_opt.py is the main script.

`python ./ppi_iterative_opt.py input_complex.pdb`

### Dependencies
RFDiffusion [https://github.com/RosettaCommons/RFdiffusion] (https://github.com/RosettaCommons/RFdiffusion)

Protein MPNN [https://github.com/dauparas/ProteinMPNN] (https://github.com/dauparas/ProteinMPNN)

Alphafold2 [https://github.com/google-deepmind/alphafold] (https://github.com/google-deepmind/alphafold)

Optional: Rosetta [https://github.com/RosettaCommons/rosetta] (https://github.com/RosettaCommons/rosetta)

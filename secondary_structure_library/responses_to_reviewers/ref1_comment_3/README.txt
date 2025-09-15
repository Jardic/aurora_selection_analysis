In this folder, I try to answer this question we got from referee 1:

3 - End of the section 'A secondary structure library': Please estimate the copy number of library RNAs with the least frequent of the combinations of the 35 mutations. This gives an idea of how reliable the sampling of all variants was in the structured library.

Here is Eds proposed structure of an answer:
We cannot do this directly because the sequencing coverage was not sufficient to see all sequences in the starting library. However, if we assume that biases at different positions during library synthesis are independent, we can calculate the expected frequency of every sequence in the starting library. This indicates that the most abundant sequence is expected to occur x-fold more frequently in the starting library than the least abundant sequence, and corresponds to an average copy number of x to y for sequences in the starting library (10^x molecules were used in the selection). Note also that, for 99% of the sequences, the most abundant sequence is expected to occur only x-fold more frequently in the starting library than the least abundant sequence.   We have highlighted this in the revised version of the manuscript. JARDA, CAN YOU PLEASE DO THIS?  

To answer this I did the following:
- Load the initial library of the secondary structure
- Compute the frequencies of every base at every positoins
- Compute the expected frequencies of every sequence in the initial pool
- Multiply these numbers by 5,6×10^13 which is what Martin Volek said is the number of molecules.

Report the minimum and maximum of this series of numbers and their ratio. 
These results are in file 'initial_lib'variance.txt'

I've also made a plot (cpm_distribution_initial_expected.pdf) which might answer what you ment in the last sentence of your answer?

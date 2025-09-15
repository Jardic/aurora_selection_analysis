In this folder, I attempt to address the Refs1 question about library overlaps (the secondary structure vs. random)

One thing he asks for are the number of overlapping positions. 
These positions are listed in the file overlapping positions, there are 15 of them. 
These are the 15 positions, where there is only one unique base that can be at a given positions in either library.

However what about those positions, where you can have multiple bases? 

To answer this better, I've put together a table called 'library_overlaps.csv', where for every position, I calculate the overlap size by looking at the number of unique bases at every position, finding out how many are overlapping and how many are there in total (union) and then dividing this intersection by the union. 

Files: 'library_design_table_ctrl.csv', 'library_design_table_strc.csv' are just the files containing the library designs in tabular form.

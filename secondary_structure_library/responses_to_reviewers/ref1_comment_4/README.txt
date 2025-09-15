In this folder, I try to answer this comment/question of referee 1:

4 - Figure 2F: Please use this figure to also show the behavior of sequences that appeared in one but not the other selection. To do this, I recommend inserting into a graph an axis corresponding to 'zero CPM' for the random library, and a corresponding axis showing the 'zero CPM' for the structured library, and plot the data points in blue and green, consistent with the blue/green scheme in the other panels. This would show how active were the sequences that were selected in one of the libraries but not the other. These two added axes would be shown in a different color. If the number of poin ts to show is too high I suggest showig one symbol (e.g. an empty circle) for every 10 or 100 sequences. This comparison is important to add, even if it turns out to be not as flattering as I expect.

To answer this, I've generated the entire encoded sequence space of the two libraries. Turned them into sets and took their intersections. This is the 'encoding overlap'. There are 128 sequences which are encoded by both libraries, they are in the file 'library_overlap_seqs.txt'. 70 of these appear in both libraries, 6 appear in the random only and 25 appear in the secondary structure library dataset only. All this is shown in the plot.

**lib_overlaps_by_generation.ipynb** is where I generate the two libraries exhaustively and take their intersection
**plot_strc_vs_ctrl_cpms.ipynb** is where I plot the results. 




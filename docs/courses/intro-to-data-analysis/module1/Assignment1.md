---
layout: basic
---

<div class="nav-links"> 
<a href=" {{ site.baseurl }}/docs/courses/intro-to-data-analysis/module1/CSV" class="prev-link">&larr; Previous Page</a> 
</div>

# Assignment 1


1.  Create instructions that can be used to start up the Anaconda prompt, navigate to the directory where you are keeping your course work, and start the jupyter notebooks you created on day one.  Also make sure you know how to create a new notebook and add it to the instructions.  

Note that the instructions will be different depending on whether you use Mac, Linux, or Windows.  On Mac and Linux the default terminal has access to the Anaconda tool.  On Windows you need to start a special Anaconda prompt. 

2.  If you haven't already create a new bookmark folder on your web browser to store class material.  Archive any links from day 1 that may be useful.

For example one link that may be useful, is the link the Python examples 

https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/Python_Tutorial.ipynb

3.  Review the Python examples in the link above.  Use it as a template to create your own Python tutorial which shows how to use, the Print statement, Math Operations, Logical Operators, Lists, Dictionaries, If/Else statements, For statements, the Import Statement and Functions.   

4.  Review the work we did Saturday on trying to find the number of points scored, date of game, and opponent for a players high scoring game.  See https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/find_high_score_basketball.ipynb

Copy the code cell by cell, and add some comments that explain what each part of the code is doing.  

5.  In the directory with the notebooks there are some new basketball related CSV files

https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/Jokic_2022_23.csv

https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/Lebron_2018_19.csv

https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/Scottie_Barnes_2022_23.csv

Download these files.  These are game logs for Nikola Jokic, Lebron James, and Scottie Barnes.  Try to restructure the code used to find the high scoring game/opponent and date for Jalen Brunson and make into a function that can be called to find the high scoring game of any player.  For example your function could take the name of the file as input, then do all the operations needed to open the file, parse the data, and return the answer.  Then use your function to get the high scoring game date and opponent for Jokic, Lebron James and Barnes. 

6.  In the directory with the notebooks there are now csv files with game logs for minor league baseball players Alan Roden, and Jackson Holliday.  

https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/Roden_2023.csv

https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/courses/intro-to-data-analysis/module1/Holliday_2023.csv


Download the files.  

Then write a Python notebook that opens each file and calculates the players batting average.  Batting average is calculated as follows

'''
batting average = sum of at bats (AB)/(sum of hits (H))
'''

Hint in the previous example we calculated the max of a column of a Dataframe.  In this example we need to calculate the sum

Bonus Points:  Both players played games at different baseball levels (Lev).  Can you calculate their batting average at each level?




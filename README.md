## Background
"Influence Island" is a mini-game within The Sims Freeplay that rewards winning players with a set of exclusive in-game items. I model this mini-game as a finite horizon Markov decision process, with the primary goal of finding the optimal set of user actions to win the full set of exclusive in-game items without spending in-game currency. 

## Current Status
I'm still making some modeling decisions to better reflect my goals! To run this (until I set up hosting), 
clone this, then run a simple user interface for decision support by using `uv run main.py` via CLI. 

## In Progress
### Must haves
* [] Empirically validate model assumptions by collecting gameplay data from tSFP social media groups
* [] Finish writeup and visualizations to justify model choices and decisions
* [] Finish conceptualizing and implementing model
* [] Create "gameplay" log collection interface
* [] Set up hosting 
* [x] Generate figures to illustrate win probability for decisions and states

### Would be nice to haves
* [] Implement simple game overlay to guide users to ideal decisions (is this allowed by EA policy? must check)
* [] Incorporate in-game currency decisions into models by defining a utility function?
* [] Permit flexible objectives such as winning a full set of items vs. just some of them

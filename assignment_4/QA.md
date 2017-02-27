**(Question 1)** *Look at the generated txt file including the ROC analysis
results. Which observer has achieved the best performance? Reader,
CAD, or Reader+CAD? What is your justification for this?*

In our case the CAD system had the best performance
1. R max FPF =  0.967.
2. R+CAD max FPF =  0.933.
3. CAD max FPF =  0.767.

The answer is quite simple: I'm not an expert radiologist and suffer from a cognitive bias that makes me weight my own judgement equally as the CAD judgement. Often I was taking the mean between CAD and my own probability which would probably increase the score of a professional but not of me as my own rating is absolute garbage.

**(Question 2)** *Looking at the sample result of the CAD system and
the ground truth in Figure 3, will CAD detection in the right lung be
considered as a true positive or a false positive in ROC analysis? Why?
Propose a modification to overcome this shortcoming. Elaborate your
answer.*

Is considered as a True Positive. ROC analysis assigns a single rating per image, ignoring information about the target/detection location. Thus, sample cases as the one shown in Figure 3 are regarded as True Positives in ROC analysis, even though the system in reality failed to detect the tumor.
A region-based analysis such as FROC analysis can aid overcome this shortcoming. FROC analysis regards a detection as a True Positive only if it is close enough to the target mark.


**(Question 3)** *As a general question far from this assignment, Com-
pare ROC and FROC analysis. Is there any situation in which FROC
provides us with some information that ROC does not?*

FROC analysis tells us about the variation that exists across case samples. It gives us information about the "difficult" cases, whereas ROC analysis is based on the overall rate of False Positives.

**(Question 4)** *What would you do to increase the power of the study?
Include more cases or more readers?*

Including more readers would make the gound truth more reliable, hence will add statistical power to the study. 
On the other hand, adding more case studies could aid the system to perform better. However, since dissagreements among readers is relatively common in the field, the whole performance assessment could be biased if the ground truth is not computed including different readers.

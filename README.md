# Rf-MSO-Wake-Model

It is a model used for wake loss analysis and AEP. 

17 july:
1. All required Python libraries for the Gaussian Wake Model have been added to the project.
2. The turbine selected for the analysis is the Vestas V110-2.0 MW.
3. All turbine specifications have been implemented in the code.
4. The only remaining turbine-related component is the power-wind curve data, which has not yet been added.

18 july:
1. the required csv files have been recieved by the coding team.
2. the power-wind curve was also given by researcher members
3. additional specifications related to the turbine are also being added such as the (rotor-area being derrived from rotor-diameter)

19 july:
1. The 3 files were added to GitHub as of now (layout, turbine, wake_model), obtained from Socrates, along with 2 CSV files (the era5_windspeed CSV is large; its sample will be put for public display, but the whole file cannot be added).
2. Many changes were made to the turbine code, which include classifying the turbine as a class and, instead of the code, giving it the actual power-wind curve for its model. (The source to be cited will be shared below.)
3. Accordingly, the assigners from the turbine model to layout and wake_model were changed so that the code does not fail, but the rest of the code is untouched.
SOURCE: https://www.thewindpower.net/turbine_power_curve_en_590_vestas_v110-2000.php

20 july:
1. The extreme weather conditions CSV was unused within the main code so a new "resilliance_testing" code was developed which utilizes the CSV alongside data gathered from the main program to visualize how different conditions effect AEP. (The results will be in the research paper)
2. A genetic algorithm was also developed to test if we could find a better version of the staggered layout, we were able to find one which has a 0.016% increase in AEP

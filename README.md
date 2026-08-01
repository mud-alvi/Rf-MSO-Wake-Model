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

21 july:
1. A staggered model in direction of the wind currents was tested to see if it was able to produce higher AEP than the predefined staggered model but it was unsucessful so it was not included in the final code.
2. In resilliance testing instead of cumilative weather testing, independent conditions such as low, moderate and high; heat, cold and gust were tested. Drought data was also gathered and was used to test weather resilliance. RESULT: Higher drought severity was associated with lower modeled weather resilience in the 2020–2025 dataset.

23 july:
1. As the AEP changes were minimal and not report worthy discussions were made on how to move forward with the current findings and actually make a reasonable discovery.

24 july:
1. Two main ideas were adopted on how to move forwards and flowcharts and all the required data was discussed.

27 july:
1. NOAA KAMA hourly observations for 2020–2025 were integrated through `weather_hazards.py`.
2. The new analysis summarizes relative humidity, recorded precipitation, trace and multi-hour precipitation flags, dust-event hours, and thunderstorm hours by year.
3. The hourly timestamps are converted from UTC to Amarillo local time so the yearly results match the local 2020–2025 calendar.
4. These variables are currently reported as hazard-exposure indicators. No arbitrary AEP-loss penalty is assigned to them.
5. Thunderstorm flags represent observed thunderstorm conditions, not measured lightning-strike counts. Humidity and precipitation alone are not sufficient for a physical icing-loss calculation; synchronized temperature and precipitation type are still required.
6. other turbine properties were also added in the 'turbine.py' file

NOAA SOURCE: https://www.ncei.noaa.gov/access/search/data-search/local-climatological-data-v2

28 July:
1. Modelled the fatigue calculation script and all the required imports were used as necessary and a validation script was also added to calculate and export the value when needed

 30 July:
 1. 

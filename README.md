# Rf-MSO-Wake-Model

It is a model used for wake loss analysis and AEP. 

17 July:
1. All required Python libraries for the Gaussian Wake Model have been added to the project.
2. The turbine selected for the analysis is the Vestas V110-2.0 MW.
3. All turbine specifications have been implemented in the code.
4. The only remaining turbine-related component is the power-wind curve data, which has not yet been added.

18 July:
1. the required csv files have been recieved by the coding team.
2. the power-wind curve was also given by researcher members
3. additional specifications related to the turbine are also being added such as the (rotor-area being derrived from rotor-diameter)

19 July:
1. The 3 files were added to GitHub as of now (layout, turbine, wake_model), obtained from Socrates, along with 2 CSV files (the era5_windspeed CSV is large; its sample will be put for public display, but the whole file cannot be added).
2. Many changes were made to the turbine code, which include classifying the turbine as a class and, instead of the code, giving it the actual power-wind curve for its model. (The source to be cited will be shared below.)
3. Accordingly, the assigners from the turbine model to layout and wake_model were changed so that the code does not fail, but the rest of the code is untouched.
SOURCE: https://www.thewindpower.net/turbine_power_curve_en_590_vestas_v110-2000.php

20 July:
1. The extreme weather conditions CSV was unused within the main code so a new "resilliance_testing" code was developed which utilizes the CSV alongside data gathered from the main program to visualize how different conditions effect AEP. (The results will be in the research paper)
2. A genetic algorithm was also developed to test if we could find a better version of the staggered layout, we were able to find one which has a 0.016% increase in AEP

21 July:
1. A staggered model in direction of the wind currents was tested to see if it was able to produce higher AEP than the predefined staggered model but it was unsucessful so it was not included in the final code.
2. In resilience testing instead of cumulative weather testing, independent conditions such as low, moderate and high; heat, cold and gust were tested. Drought data was also gathered and was used to test weather resilliance. RESULT: Higher drought severity was associated with lower modeled weather resilience in the 2020–2025 dataset.

23 July:
1. As the AEP changes were minimal and not report worthy discussions were made on how to move forward with the current findings and actually make a reasonable discovery.

24 July:
1. Two main ideas were adopted on how to move forwards and flowcharts and all the required data was discussed.

27 July:
1. NOAA KAMA hourly observations for 2020–2025 were integrated through `weather_hazards.py`.
2. The new analysis summarizes relative humidity, recorded precipitation, trace and multi-hour precipitation flags, dust-event hours, and thunderstorm hours by year.
3. The hourly timestamps are converted from UTC to Amarillo local time so the yearly results match the local 2020–2025 calendar.
4. These variables are currently reported as hazard-exposure indicators. No arbitrary AEP-loss penalty is assigned to them.
5. Thunderstorm flags represent observed thunderstorm conditions, not measured lightning-strike counts. Humidity and precipitation alone are not sufficient for a physical icing-loss calculation; synchronized temperature and precipitation type are still required.
6. other turbine properties were also added in the 'turbine.py' file

NOAA SOURCE: https://www.ncei.noaa.gov/access/search/data-search/local-climatological-data-v2

28 July:
1. specifications for the fatigue model were planned upon and the code for it was starting work
2. LCOE model was planned upon and a new fitness function was agreed upon

29 July:
1. The inputs for the fatigue model were gathered and successfully were plugged into the repository and the fatigue calculations were being tested and mean fatigue output was given
2. The LCOE model was completed which took on OPEX annual values and calculated other variables by individual calculations for more freedom in case of future changes to turbine number

31 July:
1. The fatigue model and the LCOE model were added into the GA's fitness function alongside AEP and individual penalties were set for a more resilience oriented GA.
2. The area on which the tests were being conducted was also expanded from 20Dx22.5D to 25Dx25D.
3. official Ct value of the Vestas turbine was added

1 August:
1. Instead of the ideal air density which was previously agreed upon the research team and the software decided to gather the air temperature and pressure to calculate air density of the Texas panhandle area specifically for more accurate reading.
2. the air density function was built and added into the code replacing the old ideal constant alongside that the hub height was also adjusted to match that of the Vestas model.
3. The GA model was incorporated with a better mutation rate and a tournament based GA was built to decrease the stagnation on the model. To further improve on the GA a multi-seed input was selected instead of the previous single seed simulation.
4. Previously the GA tested out different layouts only but another variable was added of yaw angle optimization to further decrease stagnation and increase chances of a potential better layout than the selected staggered baseline.

2 August:
1. 

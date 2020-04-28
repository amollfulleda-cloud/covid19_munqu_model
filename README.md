# covid19_munqu_model
Simple python class to experiment with MUNQU model for covid19 evolution in Spain. 
The mathematical model is taken from:
https://covid19.webs.upv.es/

The covid-19 data for Spain is taken from:
https://covid19.isciii.es/

1) FILES:
- covid19_model.py
- serie_historica_acumulados.csv: Data downloaded
- isciii_file:                    Folder containing downloaded data on different days
- Spain_covid_plots.html:         Example of generated picture

2) USAGE:
from a python environment run: 

import covid19_model
covid19_model.main()

It is possible to indicate the date of the reported data
covid19_model.main("2020-04-26")

3) KNOWN Issues

Analysis per region in Spain is not operative.

4) Assumption for future data:

* Simulation is run un end of June
* Start of quarentene is assumed on 2020-03-16
* End of quarentene is organized in fours steps:

1st_date = "2020-04-20" => 5% of people turns to normal life
2nd_date = "2020-05-27" => 5% of remaining people turns to normal life
3rd_date = "2020-05-02" => 10% of remaining people turns to normal life
4rd_date = "2020-05-18" => 10% of remaining people turns to normal life


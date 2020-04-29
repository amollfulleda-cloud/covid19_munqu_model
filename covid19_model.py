import pandas as pd
import numpy  as np
import datetime as dt
import wget
import os
import sys
import time
import pyswarms as ps
from scipy  import interpolate
from shutil import copyfile


import bokeh.plotting as bplot
from bokeh.io import output_file, show
from bokeh.palettes import Category10
from bokeh.layouts import gridplot, column
from bokeh.models import Div, VArea

class Covid19:
    def __init__(self, Region="SPAIN", end_sim="2020-06-30", report_date="2020-04-23", cal_date="2020-04-22", w_death=1.0, w_uci=2.0, w_hosp=0.28):

        # Define Spain Region to extract model
        self.region = Region  # Only "ALL" is working

        # Keep calibrated parameters model
        # ---------------------------------------------------------------------
        self.model_param_cal = {'beta_pre'  : 0.63,
                                'beta_post' : 0.15,
                                'delta'     : 0.9800,
                                'p1'        : 0.0309,
                                'p2'        : 0.1000,
                                'p3'        : 0.1516,
                                'p4'        : 0.5159,
                                'L0'        : 14503,
                                'I0'        : 9199}

        # ---------------------------------------------------------------------


        # Define Time Widnows for data processing ans visualization
        # ---------------------------------------------------------------------
        # Time window to take reported data
        self.start_report_date = "2020-02-27"
        self.end_report_date   = report_date

        # Define time windows for simulation and calibration
        self.start_sim_date  = "2020-02-27"
        self.end_sim_date    = end_sim

        # Time window used for model calibration
        self.start_cal_date = "2020-03-09" # No reliable data before this date
        self.end_cal_date   = cal_date     # Calibration Range

        # End of quarentene
        self.quarentene_end_1st_date = "2020-04-20"
        self.quarentene_end_2nd_date = "2020-05-27"
        self.quarentene_end_3rd_date = "2020-05-02"
        self.quarentene_end_4rd_date = "2020-05-18"

        self.quarentene_end_1st_people = 0.05
        self.quarentene_end_2nd_people = 0.05
        self.quarentene_end_3rd_people = 0.3
        self.quarentene_end_4rd_people = 0.3

        beta_pre  = self.model_param_cal['beta_pre']
        beta_post = self.model_param_cal['beta_post']

        self.quarentene_end_3rd_beta   = (beta_pre+beta_post)/2.0
        # ---------------------------------------------------------------------

        # CALIBRATION PArameters
        # ---------------------------------------------------------------------
        self.w_death = w_death
        self.w_uci   = w_uci
        self.w_hosp  = w_hosp
        # ---------------------------------------------------------------------


        # Interpolation
        # ---------------------------------------------------------------------
        # Window length for interpolation.
        # IT MUST BE AN ODD NUMBER
        self.interp_window_np = 9

        # Create Interpolation matrix
        self.M, self.K = self.get_interp_matrix(self.interp_window_np)
        # ---------------------------------------------------------------------

        # Reference data fro calibration
        # ---------------------------------------------------------------------
        self.uci_ref   = np.array([0])
        self.hosp_ref  = np.array([0])
        self.death_ref = np.array([0])
        # ---------------------------------------------------------------------


        # Define Identifiers to plot Figures
        # ---------------------------------------------------------------------
        output_file("Spain_covid_plots.html")

        self.model_plotted   = False
        self.data_plotted    = False

        # Infected people - Accumulated
        self.pIa = bplot.figure(width=480, height=320, \
                title="Accumulated No. Infected cases", \
                x_axis_label='Days', x_axis_type='datetime', \
                y_axis_label='No. cases')

        # Infected people - Daily
        self.pId = bplot.figure(width=480, height=320, \
                title="Evolution of No. Infected per day", \
                x_axis_label='Days', x_axis_type='datetime', \
                y_axis_label='No. cases')

        # Hospitalized people
        self.pHd = bplot.figure(width=480, height=320, \
                title="Evolution of No. Hospitalized per day", \
                x_axis_label='Days', x_axis_type='datetime', \
                y_axis_label='No. Cases')

        # Deaths
        self.pFa = bplot.figure(width=480, height=320, \
                title="Evolution of accumulated Deaths", \
                x_axis_label='Days', x_axis_type='datetime', \
                y_axis_label='No. Cases')

        # Recovered
        self.pRa = bplot.figure(width=480, height=320, \
                title="Evolution of No. Recovered accumulated", \
                x_axis_label='Days', x_axis_type='datetime', \
                y_axis_label='No. Cases')

        # UCI
        self.pUd = bplot.figure(width=480, height=320, \
                title="Evolution of No. UCI per day", \
                x_axis_label='Days', x_axis_type='datetime', \
                y_axis_label='No. Cases')

        # Plot information
        self.text_info = " "
        # ---------------------------------------------------------------------

    # READ_COVID_DATA_SP
    # Downloads Reported Covid-19 data from Spanish misnitry
    # and returns a dataframe with all interpolated data
    #
    def read_covid_data_sp(self):

        # Remove old data if exists
        # -----------------------------------------------------------------------------------------
        if os.path.exists('./serie_historica_acumulados.csv'):
          os.remove('./serie_historica_acumulados.csv')

        if os.path.exists('./serie_acumulados_updated.csv'):
          os.remove('./serie_acumulados_updated.csv')
        # -----------------------------------------------------------------------------------------

        # Download new data
        if self.end_report_date == dt.datetime.strftime(dt.datetime.today(), "%Y-%m-%d"):
            fdwn = wget.download('https://covid19.isciii.es/resources/serie_historica_acumulados.csv')
            print(" ")
        else:
            orig_file = "./isciii_files/"+self.end_report_date.replace("-", "")+"_serie_historica_acumulados.csv"
            dest_file = "./serie_historica_acumulados.csv"
            copyfile(orig_file, dest_file)

        # Remove commnets from last line to be able to directly load the file into pandas dataframe
        # -----------------------------------------------------------------------------------------
        fid   = open('./serie_historica_acumulados.csv', "rb")
        lines = fid.readlines()
        fid.close()

        ind = -1
        while lines[ind].decode('utf-8', 'ignore')[0:2] != 'RI':
            ind -= 1
        ind += 1

        fid_up = open('./serie_acumulados_updated.csv', "wb")
        fid_up.writelines([item for item in lines[:ind]])
        fid_up.close()
        # -----------------------------------------------------------------------------------------

        # Read CSV into a Pandas dataframe
        data_sp = pd.read_csv('./serie_acumulados_updated.csv')

        # Ensure that all column names are in lowcase and without blank spaces
        # This is to protect the script to fail when the columns name is changes with different case format
        # -----------------------------------------------------------------------------------------
        data_col  = data_sp.columns
        new_names = []
        for name_col in data_col:
            new_names.append(name_col.strip().lower())
        data_sp.columns = new_names
        # -----------------------------------------------------------------------------------------


        # make the date vector
        # -----------------------------------------------------------------------------------------
        # Check all dates are DD/MM/YYYY and convert them to YYYY-MM-DD
        for day, ind in zip(data_sp['fecha'], range(len(data_sp))):
            data_sp['fecha'].at[ind] = dt.datetime.strptime(day, '%d/%m/%Y').strftime('%Y-%m-%d')


        self.start_report_date = data_sp['fecha'].iat[0]
        self.end_report_date   = data_sp['fecha'].iat[-1]

        first_day = dt.datetime.strptime(data_sp['fecha'].iat[0], "%Y-%m-%d")
        last_day  = dt.datetime.strptime(data_sp['fecha'].iat[-1], "%Y-%m-%d")

        num_days  = (last_day - first_day).days + 1

        self.date_vec  = np.array(first_day.strftime('%Y-%m-%d'), dtype=np.datetime64) + np.arange(num_days)
        # -----------------------------------------------------------------------------------------

        # Extract accumulated data
        # -----------------------------------------------------------------------------------------
        if self.region == 'ALL':
          conf_acum  = np.zeros(num_days)
          hosp_acum  = np.zeros(num_days)
          uci_acum   = np.zeros(num_days)
          death_acum = np.zeros(num_days)
          for ind in range(num_days):
              day = first_day + dt.timedelta(days=ind)

              conf_acum[ind]  = np.sum(data_sp[data_sp['fecha'] == day.strftime('%Y-%m-%d')]['casos'])
              hosp_acum[ind]  = np.sum(data_sp[data_sp['fecha'] == day.strftime('%Y-%m-%d')]['hospitalizados'])
              uci_acum[ind]   = np.sum(data_sp[data_sp['fecha'] == day.strftime('%Y-%m-%d')]['uci'])
              death_acum[ind] = np.sum(data_sp[data_sp['fecha'] == day.strftime('%Y-%m-%d')]['fallecidos'])

              #np.sum(data_sp[data_sp['fecha'] == day.strftime("%d/%m/%Y")]['casos'])

        elif self.region == 'CT':
            conf_acum  = np.array(data_sp[data_sp['ccaa'] == 'ct']['casos'])
            hosp_acum  = np.array(data_sp[data_sp['ccaa'] == 'ct']['hospitalizados'])
            uci_acum   = np.array(data_sp[data_sp['ccaa'] == 'ct']['uci'])
            death_acum = np.array(data_sp[data_sp['ccaa'] == 'ct']['fallecidos'])
        # -----------------------------------------------------------------------------------------

        # Interpolate accumulated data,
        # Uses an cubic polynomial with a windows of 9 days, that is, 4 daya before and 4 days after.
        # -----------------------------------------------------------------------------------------


        conf_acum_interp  = self.get_interp_data(conf_acum)
        hosp_acum_interp  = self.get_interp_data(hosp_acum)
        uci_acum_interp   = self.get_interp_data(uci_acum)
        death_acum_interp = self.get_interp_data(death_acum)
        # -----------------------------------------------------------------------------------------

        # Construct Dataframe
        # -----------------------------------------------------------------------------------------
        data_dict = {'conf_acum' : conf_acum,
                     'hosp_acum' : hosp_acum,
                     'uci_acum'  : uci_acum,
                     'death_acum': death_acum,
                     'conf_acum_interp' : conf_acum_interp,
                     'hosp_acum_interp' : hosp_acum_interp,
                     'uci_acum_interp'  : uci_acum_interp,
                     'death_acum_interp': death_acum_interp,
                     }
        self.df = pd.DataFrame(data=data_dict, index=self.date_vec)
        # -----------------------------------------------------------------------------------------


        # Generate Reference data for Calidation
        # -----------------------------------------------------------------------------------------
        self.uci_ref   = self.df['uci_acum_interp'][  self.start_cal_date:self.end_cal_date]
        self.hosp_ref  = self.df['hosp_acum_interp'][ self.start_cal_date:self.end_cal_date]
        self.death_ref = self.df['death_acum_interp'][self.start_cal_date:self.end_cal_date]
        # -----------------------------------------------------------------------------------------



    # RUN_MODEL:
    #   Implements the epidemic model developed by MUNQU
    #   Run the model given a set of invot vector
    #   Return the hospitalized, the UCI and Deaths
    #   INOUT Vector:
    #
    #      model_param: Structure with the following fields: I0, H0, Beta, gamma1, p1, p2, p3, p4, L0, I0
    #      start_date_str: Day from which the simulation starts reuslts are given
    #      end_date_str:   Last day of the model simulation
    def run_model(self, model_param=None, start_date_str=None, end_date_str=None):

        # Initialize parameters
        if model_param == None:
            model_param = self.model_param_cal
        if start_date_str == None:
            start_date_str = self.start_sim_date
        if end_date_str == None:
            end_date_str = self.end_sim_date


        # Model parameter definition
        # -----------------------------------------------------
        beta_pre  = model_param['beta_pre']
        beta_post = model_param['beta_post']
        delta     = model_param['delta']
        p1        = model_param['p1']
        p2        = model_param['p2']
        p3        = model_param['p3']
        p4        = model_param['p4']

        gamma1    = 1/5.2
        gamma2    = p1/5.8
        gamma3    = p3/1
        alpha1    = (1-p1)/14
        alpha2    = (1-p2-p3)/7
        alpha3    = (1-p4)/14
        d1        = p2/7.5
        d2        = p4/8
        nu        = 1/6

        PT        = 47100396

        #first_day_str       = '2020-02-27'
        first_day_str       = start_date_str
        last_day_str        = end_date_str
        # -----------------------------------------------------


        # Model equations
        # -----------------------------------------------------
        num_days  = (dt.datetime.strptime(last_day_str, "%Y-%m-%d") - dt.datetime.strptime(first_day_str, "%Y-%m-%d")).days + 1

        S_vec     = np.zeros(num_days)
        Q_vec     = np.zeros(num_days)
        L_vec     = np.zeros(num_days)
        I_vec     = np.zeros(num_days)
        H_vec     = np.zeros(num_days)
        U_vec     = np.zeros(num_days)
        HU_vec    = np.zeros(num_days)
        R_vec     = np.zeros(num_days)
        F_vec     = np.zeros(num_days)

        date_vec  = np.array(first_day_str, dtype=np.datetime64) + np.arange(num_days)

        L_vec[0] = model_param['L0']
        I_vec[0] = model_param['I0']
        S_vec[0] = PT - L_vec[0] - I_vec[0]


        # Define beta and delta vectos along days
        # Alarm declared on "2020-03-16"
        alarm_start_date = "2020-03-16"
        beta_vec  = np.ones(num_days)*beta_pre
        beta_vec[ date_vec >= np.datetime64(alarm_start_date) ]             = beta_post
        beta_vec[ date_vec >= np.datetime64(self.quarentene_end_3rd_date) ] = self.quarentene_end_3rd_beta

        delta_vec  = np.zeros(num_days)
        delta_vec[ date_vec == np.datetime64(alarm_start_date) ] = delta

        tau_vec    = np.zeros(num_days)
        tau_vec[date_vec == np.datetime64(self.quarentene_end_1st_date)] = self.quarentene_end_1st_people
        tau_vec[date_vec == np.datetime64(self.quarentene_end_2nd_date)] = self.quarentene_end_2nd_people
        tau_vec[date_vec == np.datetime64(self.quarentene_end_3rd_date)] = self.quarentene_end_3rd_people
        tau_vec[date_vec == np.datetime64(self.quarentene_end_4rd_date)] = self.quarentene_end_4rd_people

        # Run model
        for ind, day in zip(np.arange(num_days-1), date_vec):

            # Read curren values
            S_c  = S_vec[ind]
            Q_c  = Q_vec[ind]
            L_c  = L_vec[ind]
            I_c  = I_vec[ind]
            H_c  = H_vec[ind]
            U_c  = U_vec[ind]
            HU_c = HU_vec[ind]
            R_c  = R_vec[ind]
            F_c  = F_vec[ind]

            # Compute next value
            S_next  = S_c  - beta_vec[ind]*S_c*I_c/PT - delta_vec[ind]*S_c + tau_vec[ind]*Q_c
            Q_next  = Q_c  + delta_vec[ind]*S_c - tau_vec[ind]*Q_c
            L_next  = L_c  + beta_vec[ind]*S_c*I_c/PT - gamma1*L_c
            I_next  = I_c  + gamma1*L_c - (gamma2+alpha1)*I_c
            H_next  = H_c  + gamma2*I_c - (d1 + alpha2 + gamma3)*H_c
            U_next  = U_c  + gamma3*H_c - (d2 + alpha3)*U_c
            HU_next = HU_c + alpha3*U_c - nu*HU_c
            R_next  = R_c  + alpha1*I_c + alpha2*H_c + nu*HU_c
            F_next  = F_c  + d1*H_c + d2*U_c

            # Update next values
            S_vec[ind+1]  = S_next
            Q_vec[ind+1]  = Q_next
            L_vec[ind+1]  = L_next
            I_vec[ind+1]  = I_next
            H_vec[ind+1]  = H_next
            U_vec[ind+1]  = U_next
            HU_vec[ind+1] = HU_next
            R_vec[ind+1]  = R_next
            F_vec[ind+1]  = F_next

        return L_vec, I_vec, H_vec, U_vec, F_vec, R_vec, HU_vec, date_vec

    # munqu_cost_function
    def munqu_cost_function(self, x):

        numx = np.shape(x)[0]

        fret = np.zeros(numx)

        for ind in range(numx):

          xb = x[ind,:]

          model_param = {
              'beta_pre'  : xb[0],
              'beta_post' : xb[1],
              'delta'     : xb[2],
              'p1'        : xb[3],
              'p2'        : xb[4],
              'p3'        : xb[5],
              'p4'        : xb[6],
              'L0'        : xb[7]*100000,
              'I0'        : xb[8]*100000}


          # Run Model: from START_SIM to END_CAL
          lr, ir, hr, ur, dr, rr, hur, dtr = self.run_model(model_param, self.start_sim_date, self.end_cal_date)

          # Take only measurable range
          dvm = dr[(dtr >= np.datetime64(self.start_cal_date)) & (dtr <= np.datetime64(self.end_cal_date))]
          uvm = ur[(dtr >= np.datetime64(self.start_cal_date)) & (dtr <= np.datetime64(self.end_cal_date))]
          hvm = hr[(dtr >= np.datetime64(self.start_cal_date)) & (dtr <= np.datetime64(self.end_cal_date))]
          ivm = ir[(dtr >= np.datetime64(self.start_cal_date)) & (dtr <= np.datetime64(self.end_cal_date))]

          # Considering accumulated New HOSP cases
          gamma2 = xb[3]/5.8
          hv     = np.cumsum(gamma2*ivm)

          # Considering accumulated New UCI cases
          gamma3 = (1 - xb[6])/14
          uv     = np.cumsum(gamma3*hvm)

          # For Deaths the reported data is already accumulated data
          dv     = dvm

          # Evaluate error Function
          uref = self.uci_ref
          dref = self.death_ref
          href = self.hosp_ref

          wu   = self.w_uci   #0.0#1/2.1
          wd   = self.w_death #1.0#1/2.1
          wh   = self.w_hosp  #0.0#wu/10

          ferror   = wd*sum(abs(dref - dv)*abs(dref))/sum(pow(dref, 2))
          ferror  += wu*sum(abs(uref - uv)*abs(uref))/sum(pow(uref, 2))
          ferror  += wh*sum(abs(href - hv)*abs(href))/sum(pow(href, 2))

          fret[ind] = ferror

        return fret

    def get_interp_matrix(self, wlen):
        wpre = int(np.floor(wlen/2))

        M = np.matrix([[pow(j, i) for i in range(0, 4)] for j in range(-wpre, wpre+1)])

        K = np.linalg.inv(M.T*M)*M.T

        return M, K

    def interp_data(self, dv, wlen, M, K):
        #
        # days (x):      -2    -1     0     1     2
        #
        # Cubic interpolator: p(x) = a0 + a1*x + a2*x^2 + a3*x^3
        #
        #       p(-2)  = a0  +  a1*(-2)  +  a2*(-2)^2  + a3*(-2)^3
        #       p(-1)  = a0  +  a1*(-1)  +  a2*(-1)^2  + a3*(-1)^3
        #       p( 0)  = a0  +  a1*( 0)  +  a2*( 0)^2  + a3*( 0)^3
        #       p( 1)  = a0  +  a1*( 1)  +  a2*( 1)^2  + a3*( 1)^3
        #       p( 2)  = a0  +  a1*( 2)  +  a2*( 2)^2  + a3*( 2)^3
        #
        #       pv     = M*av
        #
        #       M = matrix([[1, -2, 4, -8], [1, -1, 1, -1], [1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8]])
        #
        #       E2 = (pv - dv)'*(pv - dv) = (M*av - dv)'*(M*av - dv) =
        #       E2 = av'*M'*M*av - av'*M'*dv - dv'*M*av + dv'dv
        #       DIFF(E2) = 0 =>
        #       2(M'*M)*av - 2*M'*dv = 0 =>
        #
        #       av = K*dv
        #       K  = INV(M'*M)*M'
        #
        #       p(x) = xv'*av where
        #       xv   = [1 x x^2 x^3]
        #
        # Given an input colunm vector DV and assuming the data corresponds
        # to positions [-2, -1, 0, 1, 2], returns a colunm vector to these positions
        #
        #

        av = K*(np.matrix(dv).T)

        iv = M*av

        return np.array(iv).reshape(wlen)

    def get_interp_data(self, data_vec):
        # -----------------------------------------------------------------
        # GET_INTERP_DATA: Interpolates DATA_VEC and returns the interpolated data.
        #                  Curbic interpolation is used. 5-sample window sweeps DATA_VEC
        #                  and interpolates best fitting of central sample. First two samples and
        #                  last two samples of DATA_VEC are taken from first and last windows.
        #                  WLEN: Windows Lengths. It must be a n ODD number.
        # -----------------------------------------------------------------

        # Widnow Length
        wlen = self.interp_window_np

        # Data length
        npoints = len(data_vec)

        ccv     = np.zeros(npoints)


        wpre = int(np.floor(wlen/2))

        # First two slamples
        dv = data_vec[0:wlen]
        iv = self.interp_data(dv, wlen, self.M, self.K)

        ccv[0:wpre] = iv[0:wpre]

        # Middle samples
        for ind in range(wpre+1, npoints-wpre):
            dv = data_vec[ind-wpre:ind+wpre+1]
            iv = self.interp_data(dv, wlen, self.M, self.K)

            ccv[ind] = iv[wpre]

        # Last two samples
        dv = data_vec[-wlen:]
        iv = self.interp_data(dv, wlen, self.M, self.K)

        ccv[-wpre-1:] = iv[-wpre-1:]

        return ccv

    # PLOT_MODEL
    # Given a MODEL_PARAM and START_DATE and END_DATE
    # By default uses the Simulation dates for plots
    # Runs the equivalent model and update plots for Infected, Hospitalized, at UCI, Deaths and Recovered.
    # NOTE: When plotting DATA and MODEL model plt MUSt be called FIRST !!!!!!
    def plot_model(self, model_param=None, start_date=None, end_date=None):


        # Initialize parameters
        if model_param == None:
            model_param = self.model_param_cal
        if start_date == None:
            start_date = self.start_sim_date
        if end_date == None:
            end_date = self.end_sim_date


        # Run Model
        # -----------------------------------------------------
        L_vec, I_vec, H_vec, U_vec, F_vec, R_vec, HU_vec, date_vec = self.run_model(model_param, start_date, end_date)

        # Check wether reported data has been plotted before
        # -----------------------------------------------------
        if self.data_plotted == True:
            print(" ")
            print("WARNING: Reported data ploted before Model data. ")
            print("         Bar Width of reported data might not be well fitted")
            print(" ")

        # Plot Figures
        # -----------------------------------------------------
        # Acumulados Infectados
        self.pIa.line(x=date_vec, y=np.cumsum(L_vec)/5.2, line_width=2, line_color="orange", legend_label="Model")

        # Total Infectados nuevos por dia
        self.pId.line(x=date_vec, y=I_vec, line_width=2, line_color="orange", legend_label="Model")

        # Total Hospitalizados por día
        gamma2 = model_param['p1']/5.8
        self.pHd.line(x=date_vec, y=np.cumsum(I_vec*gamma2), line_width=2, line_color='orange', legend_label="Model")

        # Total Fallecidos por día
        self.pFa.line(x=date_vec, y=F_vec, line_width=2, line_color='orange', legend_label="Model")

        # Total Recuperados por día
        self.pRa.line(x=date_vec, y=R_vec, line_width=2, line_color='green')

        # Total en UCI por día
        gamma3 = (1 - model_param['p4'])/14
        self.pUd.line(x=date_vec, y=np.cumsum(H_vec*gamma3), line_width=2, line_color='orange', legend_label="Model")

        # Show all plots
        self.text_info = "<h1>Simulated evolution of COVID-19 on Spain</h1>\
        <h2>Calibration weights: DEATHS: %.2f, UCI: %.2f, HOSP: %.2f </h2>\
        <strong>Pandemia starts on:                  </strong> 27th Feb 2020                        <br>\
        <strong>start of Quarentine:                 </strong> 16th March 2020                      <br>\
        <strong>End of Quarentine:                   </strong> 2nd May 2020                         <br>\
        <strong>Transmisivity rate before quarentine:</strong> <var>Beta<sub>pre</sub>  = %.2f</var><br>\
        <strong>Transmisivity rate after quarentine: </strong> <var>Beta<sub>post</sub> = %.2f</var><br>\
        <strong>%% of population on Quarentine:      </strong> <var>Delta = %.2f%%</var>            <br> \
        <strong>%% of infected people hospitalized:  </strong> <var>p1    = %.2f%%</var>            <br> \
        <strong>%% of deaths in hospitalzed patients:</strong> <var>p2    = %.2f%%</var>            <br> \
        <strong>%% of hospitalzed that goes to UCI:  </strong> <var>p3    = %.2f%%</var>            <br> \
        <strong>%% of death in UCI patients:         </strong> <var>p4    = %.2f%%</var>            <br> \
        <strong>%% Initial No. exposed people:       </strong> <var>L(0)  = %.d</var>               <br> \
        <strong>%% Initial No. infected people:      </strong> <var>I(0)  = %.d</var>               <br>" \
                              %(self.w_death, \
                                self.w_uci, \
                                self.w_hosp, \
                                model_param['beta_pre'], \
                                model_param['beta_post'], \
                                model_param['delta']*100, \
                                model_param['p1']*100, \
                                model_param['p2']*100, \
                                model_param['p3']*100, \
                                model_param['p4']*100, \
                                model_param['L0'], \
                                model_param['I0'])

        # Indicate ploted data is done
        self.model_plotted = True


    # PLOT_REPORTED_DATE
    # Given a the DataFrame after read data and START_DATE and END_DATE
    # By Default uses Start_Report_DAte and End_report_date
    # plots for Infected, Hospitalized, at UCI, Deaths and Recovered.
    # NOTE: When plotting REPORTED DATA and MODEL model plt MUSt be called FIRST !!!!!!
    def plot_reported_data(self, AddInterp=False):

        ## Initialize parameters
        #if start_date == None:
        #    start_date = self.start_report_date
        #if end_date == None:
        #    end_date = self.end_report_date

        # Compute Vertical Bar width
        # -----------------------------------------------------
        # DAta is reported per days and bokeh vBAR uses usecs as units to plot x_axis in datetime format
        var_width = 3600*24*1000

        # Get reported model in measurable interval
        # -----------------------------------------------------
        report_start_date = np.datetime64(self.start_report_date)
        report_end_date   = np.datetime64(self.end_report_date)
        report_num_days   = int((report_end_date - report_start_date+1)/np.timedelta64(1, 'D'))

        date_meas = np.array(report_start_date, dtype=np.datetime64) + np.arange(report_num_days)
        #date_meas = date_vec[(date_vec >= np.datetime64(start_date)) & (date_vec <= np.datetime64(end_date))]
        cases_rep = self.df['conf_acum'][report_start_date:report_end_date]
        death_rep = self.df['death_acum'][report_start_date:report_end_date]
        hosp_rep  = self.df['hosp_acum'][report_start_date:report_end_date]
        uci_rep   = self.df['uci_acum'][report_start_date:report_end_date]

        # Get Interpolated data is requestes
        if AddInterp == True:
            cases_rep_interp = self.df['conf_acum_interp'][report_start_date:report_end_date]
            death_rep_interp = self.df['death_acum_interp'][report_start_date:report_end_date]
            hosp_rep_interp  = self.df['hosp_acum_interp'][report_start_date:report_end_date]
            uci_rep_interp   = self.df['uci_acum_interp'][report_start_date:report_end_date]

        # Plot Figures
        # -----------------------------------------------------
        # Acumulados Infectados
        self.pIa.vbar(x=date_meas, top=cases_rep, width=var_width, fill_alpha=0.7, line_color="white", legend_label="Reported data")
        if AddInterp == True:
            self.pIa.line(x=date_meas, y=cases_rep_interp, line_width=2, line_color="blue", legend_label="Reported Interpolated")

        # Total Hospitalizados por día
        self.pHd.vbar(x=date_meas, top=hosp_rep, width=var_width, fill_alpha=0.7, line_color="white", legend_label="Reported data")
        if AddInterp == True:
            self.pHd.line(x=date_meas, y=hosp_rep_interp, line_width=2, line_color="blue", legend_label="Reported Interpolated")


        # Total Fallecidos por día
        self.pFa.vbar(x=date_meas, top=death_rep, width=var_width, fill_alpha=0.7, line_color="white", legend_label="Reported data")
        if AddInterp == True:
            self.pFa.line(x=date_meas, y=death_rep_interp, line_width=2, line_color="blue", legend_label="Reported Interpolated")

        # Total Recuperados por día

        # Total en UCI por día
        self.pUd.vbar(x=date_meas, top=uci_rep, width=var_width, fill_alpha=0.7, line_color="white", legend_label="Reported data")
        if AddInterp == True:
            self.pUd.line(x=date_meas, y=uci_rep_interp, line_width=2, line_color="blue", legend_label="Reported Interpolated")

        # Indicate reported data is done
        self.data_plotted = True


    def show_plots(self):

        self.pIa.legend.location="top_left"
        self.pId.legend.location="top_left"
        self.pHd.legend.location="top_left"
        self.pFa.legend.location="top_left"
        self.pUd.legend.location="top_left"

        show(column(Div(text=self.text_info), gridplot([self.pHd, self.pUd, self.pFa, self.pId, self.pIa], ncols=2)))

    def swarm_cal(self):

        # NOTE: Bounds in Beta_PRE are calculated calibrating deaths until 20/3/2020
        max_bound = np.ones(9)
        max_bound[0] = 0.8
        max_bound[1] = 0.3
        max_bound[2] = 0.98
        max_bound[3] = 0.2
        max_bound[4] = 0.2
        max_bound[5] = 0.2
        max_bound[6] = 0.7
        max_bound[7] = 0.2
        max_bound[8] = 0.2

        min_bound = np.zeros(9)
        min_bound[0] = 0.5
        min_bound[1] = 0.05
        min_bound[2] = 0.1
        min_bound[3] = 0.01
        min_bound[4] = 0.01
        min_bound[5] = 0.01
        min_bound[6] = 0.01
        min_bound[7] = 0.05
        min_bound[8] = 0.05

        bounds=(min_bound,max_bound)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        optimizer = ps.single.GlobalBestPSO(n_particles=20, \
                                            dimensions=9, \
                                            options=options, \
                                            bounds=bounds, \
                                            bh_strategy = "periodic", \
                                            vh_strategy = "invert")

        cost, pos = optimizer.optimize(self.munqu_cost_function, \
                                       iters = 1000)

        self.model_param_cal = {'beta_pre'  : pos[0],
                                'beta_post' : pos[1],
                                'delta'     : pos[2],
                                'p1'        : pos[3],
                                'p2'        : pos[4],
                                'p3'        : pos[5],
                                'p4'        : pos[6],
                                'L0'        : pos[7]*100000,
                                'I0'        : pos[8]*100000}

    # MODEL_SENSITIVITY
    # Given a model defined on MODEL_PARAM
    # moves each parameter +/-5% and extract the lower
    # and upper bound of each curve of itereset:
    # Infected, Hospitalized, at UCI, Deaths and Recovered.
    def model_sensitivity(self):

        beta_pre_vec  = [self.model_param_cal['beta_pre']*0.95,  self.model_param_cal['beta_pre']*1.05]
        beta_post_vec = [self.model_param_cal['beta_post']*0.95, self.model_param_cal['beta_post']*1.05]
        delta_vec     = [self.model_param_cal['delta']*0.95,     self.model_param_cal['delta']*1.05]
        p1_vec        = [self.model_param_cal['p1']*0.95,  self.model_param_cal['p1']*1.05]
        p2_vec        = [self.model_param_cal['p2']*0.95,  self.model_param_cal['p2']*1.05]
        p3_vec        = [self.model_param_cal['p3']*0.95,  self.model_param_cal['p3']*1.05]
        p4_vec        = [self.model_param_cal['p4']*0.95,  self.model_param_cal['p4']*1.05]
        L0_vec        = [self.model_param_cal['L0']*0.95,  self.model_param_cal['L0']*1.05]
        I0_vec        = [self.model_param_cal['I0']*0.95,  self.model_param_cal['I0']*1.05]


        # Get reported model in measurable interval
        # -----------------------------------------------------
        start_date = np.datetime64(self.start_sim_date)
        end_date   = np.datetime64(self.end_sim_date)
        num_days   = int((end_date - start_date+1)/np.timedelta64(1, 'D'))

        date_vec = np.array(start_date, dtype=np.datetime64) + np.arange(num_days)
        vlen     = len(date_vec)

        npoints  = pow(2, 9)
        ir_mat   = np.zeros((npoints, vlen))

        l_mat    = np.zeros((npoints, vlen))
        i_mat    = np.zeros((npoints, vlen))
        h_mat    = np.zeros((npoints, vlen))
        u_mat    = np.zeros((npoints, vlen))
        d_mat    = np.zeros((npoints, vlen))
        r_mat    = np.zeros((npoints, vlen))
        hu_mat   = np.zeros((npoints, vlen))
        ind      = 0
        for beta_pre_val in beta_pre_vec:
            for beta_post_val in beta_post_vec:
                for delta_val in delta_vec:
                    for p1_val in p1_vec:
                        for p2_val in p2_vec:
                            for p3_val in p3_vec:
                                for p4_val in p4_vec:
                                    for L0_val in L0_vec:
                                        for I0_val in I0_vec:
                                            model_param = {
                                                'beta_pre'  : beta_pre_val  ,
                                                'beta_post' : beta_post_val  ,
                                                'delta'     : delta_val  ,
                                                'p1'        : p1_val,
                                                'p2'        : p2_val,
                                                'p3'        : p3_val,
                                                'p4'        : p4_val,
                                                'L0'        : L0_val,
                                                'I0'        : I0_val}


                                            lr, ir, hr, ur, dr, rr, hur, dtr = self.run_model(model_param, \
                                                                                              self.start_sim_date, \
                                                                                              self.end_sim_date)

                                            l_mat[ind][:]  = lr
                                            i_mat[ind][:]  = ir
                                            h_mat[ind][:]  = hr
                                            u_mat[ind][:]  = ur
                                            d_mat[ind][:]  = dr
                                            r_mat[ind][:]  = rr
                                            hu_mat[ind][:] = hur

                                            # Update counter
                                            ind += 1

        lr_min  = l_mat.min(axis=0)
        ir_min  = i_mat.min(axis=0)
        hr_min  = h_mat.min(axis=0)
        ur_min  = u_mat.min(axis=0)
        dr_min  = d_mat.min(axis=0)
        rr_min  = r_mat.min(axis=0)
        hur_min = hu_mat.min(axis=0)

        lr_max  = l_mat.max(axis=0)
        ir_max  = i_mat.max(axis=0)
        hr_max  = h_mat.max(axis=0)
        ur_max  = u_mat.max(axis=0)
        dr_max  = d_mat.max(axis=0)
        rr_max  = r_mat.max(axis=0)
        hur_max = hu_mat.max(axis=0)

        # Acumulados Infectados
        self.pIa.varea(x=date_vec, y1=np.cumsum(lr_min)/5.2, y2=np.cumsum(lr_max)/5.2, fill_color="orange", fill_alpha=0.2)

        # Total Infectados nuevos por dia
        self.pId.varea(x=date_vec, y1=ir_min, y2=ir_max, fill_color="orange", fill_alpha=0.2)

        # Total Hospitalizados por día
        gamma2 = self.model_param_cal['p1']/5.8
        self.pHd.varea(x=date_vec, y1=np.cumsum(ir_min*gamma2), y2=np.cumsum(ir_max*gamma2), fill_color='orange', fill_alpha=0.2)

        # Total Fallecidos por día
        self.pFa.varea(x=date_vec, y1=dr_min, y2=dr_max, fill_color="orange", fill_alpha=0.2)

        # Total Recuperados por día
        self.pRa.varea(x=date_vec, y1=rr_min, y2=rr_max, fill_color="orange", fill_alpha=0.2)

        # Total en UCI por día
        gamma3 = (1 - self.model_param_cal['p4'])/14
        self.pUd.varea(x=date_vec, y1=np.cumsum(hr_min*gamma3), y2=np.cumsum(hr_max*gamma3), fill_color='orange', fill_alpha=0.2)


# Main function to organise the data processign and plotting
def main(Region="ALL", ReportDate="Today", CalDate=None, CalModel=None):

    # Check Report Date
    if ReportDate == "Today":
        day_report = dt.datetime.today()
        report_dt  = dt.datetime.strftime(day_report, "%Y-%m-%d")
    else:
        day_report = dt.datetime.strptime(ReportDate, "%Y-%m-%d")
        report_dt  = ReportDate

    # Default califration day is two days before report date
    if CalDate == None:
        day_cal = day_report - dt.timedelta(2)
        cal_dt  = dt.datetime.strftime(day_cal, "%Y-%m-%d")

    # Create the Covid Class object
    covid = Covid19(Region=Region, \
                    end_sim    = "2020-06-27", \
                    report_date= report_dt, \
                    cal_date   = cal_dt, \
                    w_death    = 1.0, \
                    w_uci      = 2.20, \
                    w_hosp     = 0.30)

    # Read Reported data
    covid.read_covid_data_sp()

    # Calibrate Model
    if CalModel == "Yes":
        covid.swarm_cal()

    # Run Model With calibrated parameters
    covid.run_model()

    # Sensitivity analysis
    covid.model_sensitivity()

    # Plot Model
    covid.plot_model()
    covid.plot_reported_data(AddInterp=True)
    covid.show_plots()


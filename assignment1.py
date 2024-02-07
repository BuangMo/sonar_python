import numpy as np
import pyaudio, queue, time, wave
from math import ceil
import matplotlib.pyplot as plt
from threading import Thread
from scipy.signal import hilbert, chirp

class PulseRadar:
    """Transmits and captures pulse data for object detection"""
    # declaration of class variables
    Fs = 48000.0                                    # sampling rate
    Ts = 1/Fs                                       # time between samples
    chunk = 48000
    volume = 1.0                                    # volume of transmitted signal
    duration = 2.0                                  # duration of the signal
    sound_speed = 343.0                             # mitres per second
    
    def __init__( self, pulses, start_freq, max_dist, min_dist, bandwidth ):
        self.num_pulses = pulses                    # the number of pulses in the signal
        self.fc = start_freq                        # the frequency of the carrier signal
        self.R_max = max_dist                       # the maximum distance that the radar should detect objects
        self.R_min = min_dist                       # range resolution as given in the guide
        self.rcv = []                               # recorded echo data
        self.bandwidth = bandwidth                  # bandwidth of the pulse

        err = pyaudio.PyAudio().terminate();
        #if( err != pyaudio.PyAudio().paNoError ) goto error;
        
        # pyAudio params
        self.p = pyaudio.PyAudio()                  # Create an interface to PortAudio
        self.format = pyaudio.paFloat32             # sets the format of the data to float
        self.channel = 1                            # sets the number of channels

    def generate_signal(self):
        '''Generates the linear frequency modulated signal'''
        modulated = []

        pulse_width = (2 * self.R_min) / PulseRadar.sound_speed
        PRI = (2 * self.R_max) / PulseRadar.sound_speed

        # calculate signal parameters
        self.lstn_samples = ceil((PRI - pulse_width) / PulseRadar.Ts)   #**** might be wrong
        self.pw_samples = ceil(pulse_width / PulseRadar.Ts)
        self.total_samples = self.pw_samples + self.lstn_samples
        silence = (PulseRadar.duration * PulseRadar.Fs-self.num_pulses * (self.pw_samples + self.lstn_samples)) / 2

        print(f"The pulse signal has {self.pw_samples} pulse width samples and {self.lstn_samples} dead time samples.")       

        # construction of the signals
        zeros_samples = np.zeros(self.lstn_samples)     # the number of samples to listen between pulses
        delay_samples = np.zeros(int(silence))          # the silence before and after pulses

        # carrier signal and template   
        pulse = chirp(
            np.linspace(0, pulse_width, self.pw_samples), 
            f0=self.fc,
            f1=self.fc + self.bandwidth, 
            t1=pulse_width, 
            method='linear'
        ).astype(np.float32)

        # message signal
        modulated = np.append(modulated, delay_samples)
        for i in range(self.num_pulses):
            modulated = np.append(modulated, pulse)
            modulated = np.append(modulated, zeros_samples)

        modulated = np.append(modulated, delay_samples)
        modulated = np.array(modulated).astype(np.float32)

        # plot the time and frequency domain signals (optional)
        #self.plotter(modulated)

        # modulating the message signal with the carrier
        self.output_bytes = (PulseRadar.volume * modulated).tobytes()

        # generate the pulse template
        upper_lim = len(delay_samples) + self.pw_samples
        self.template = modulated[len(delay_samples) - 1:upper_lim]

    def plotter(self, f_x, plot_sel=2):
        '''Plots the time series and frequency spectrums of the given signal'''
        # performs the DFT of the given signal
        F_X = np.fft.fftshift(np.fft.fft(f_x, 1024))

        # defines the time and freq range of the signal
        freq = np.arange(-PulseRadar.Fs/2, PulseRadar.Fs/2, PulseRadar.Fs / len(F_X))
        t = np.arange(len(f_x)) / PulseRadar.Fs

        # plots the time and frequency spectrum
        if plot_sel == 1:
            plt.subplot(111)
            plt.plot(t, f_x)
            plt.xlabel("time (s)")
            plt.ylabel('Magnitude')
            plt.title('LFM time-domain Signal')
            plt.grid(True)
        else:
            _, ax = plt.subplots(2, 1)
            ax[0].plot(t, f_x)
            ax[0].set(xlabel="time (s)", ylabel='Magnitude', title='LFM time-domain Signal')
            ax[0].grid(True)

            ax[1].plot(freq, np.abs(F_X))
            ax[1].set(xlabel="Frequency (kHz)", ylabel='Magnitude', title='LFM frequency-domain Signal')
            ax[1].grid(True)

        plt.show()

    def emitter(self):
        try:
            ostream = self.p.open(
                format=self.format,
                channels=self.channel,
                rate=int(PulseRadar.Fs),
                output=True
            )

            start_time = time.time()
            print( "Playing..." )
            ostream.write( self.output_bytes )
            print("Played sound for {:.2f} seconds".format(time.time() - start_time))

            # Stop and Close the output stream
            ostream.stop_stream()
            ostream.close()
        except (OSError, NameError, ValueError) as error:
            self.p.terminate()
            print(f'A {type(error).__name__} has occured')

    def listener(self):
        frames = []
        Qin = queue.Queue()
        
        try:
            istream = self.p.open(
                format=self.format,
                channels=self.channel,
                rate=int(PulseRadar.Fs),
                input=True
            )

            print("Recording...")

            # captures the data
            for dt in range(0, int((PulseRadar.duration * PulseRadar.Fs)/(PulseRadar.chunk))):
                data = istream.read(PulseRadar.chunk)
                frames = np.frombuffer(data, dtype=np.float32)
                Qin.put(frames)

            # Stop and Close the input stream
            istream.stop_stream()
            istream.close()

            # capture the recorded data in a list
            while(not Qin.empty()):
                rcv_data = Qin.get()
                self.rcv = np.append(self.rcv, rcv_data)
        except (OSError, NameError, ValueError) as error:
            self.p.terminate()
            print(f'A {type(error).__name__} has occured')

    def savefile(self):
        filename = "output.wave"
        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channel)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(PulseRadar.Fs)
        wf.writeframes(b''.join(self.rcv))
        wf.close()

    def terminate_pa( self ):
        """Terminates the pyaudio binding to portaudio"""
        self.p.terminate()
    
    def pulse_compression(self):
        print("Applying the matched filter")
        self.plotter(self.rcv, 1)
        matched_filtered = np.convolve(hilbert(self.rcv), np.conj(np.flipud(hilbert(self.template))), mode='same')
        #matched_filtered = fftconvolve(hilbert(self.rcv), np.conj(hilbert(self.template[::-1])), mode='same')
        print( "Maximum peak at: " + str( matched_filtered.argmax()))
        self.mfiltered = matched_filtered
        self.plotter(self.mfiltered.real, 1)

    def data_matrix( self ):
        print(f"Arrange the data into a matrix of {self.num_pulses} pulses.")
        # get the start of the first fast time sampling point
        origin_smpl = int(self.mfiltered[:27100].argmax())
        print(f'max at {self.mfiltered[:27100].argmax()}')

        #gets the data into a matrix of complex numbers
        sampled = self.lstn_samples+self.pw_samples
        self.pri_data = np.zeros((sampled, self.num_pulses), dtype='complex_')
        starts, stops = 0, 0
        
        for i in range(self.num_pulses):
            starts = origin_smpl+i*sampled
            stops = origin_smpl+(i+1)*sampled
            self.pri_data[:,i] = self.mfiltered[starts:stops]
            #print(f"{i}. Start={starts} stop={stops} origin={origin_smpl} difference={stops-starts}.")

        #integrate the pulses
        dt = np.mean(self.pri_data.T, axis=0)
        analytic_sigg = np.abs(dt)
        smps = np.arange( dt.size )

        #plt.subplot(311)
        plt.plot( smps, analytic_sigg )
        plt.xlabel( 'samples' )
        plt.ylabel( 'amplitude' )
        plt.grid()

        #plt.subplot(312)
        #plt.plot(np.arange(self.rcv.size), self.rcv)
        #plt.xlabel( 'samples' )
        #plt.ylabel( 'amplitude' )
        #plt.grid()

        #plt.subplot(313)
        #plt.plot( np.arange( self.mfiltered.size ), np.real(self.mfiltered))
        #plt.xlabel( 'samples' )
        #plt.ylabel( 'amplitude' )
        #plt.grid()
        #plt.show()
        
    def doppler_processing(self):
        print( "Applying the Doppler Processing to the data matrix" )
        
        # calculate range and Doppler resolutions
        delta_r = PulseRadar.sound_speed/(2*2e3)
        delta_v = PulseRadar.sound_speed/(2*9e3*(self.pw_samples*PulseRadar.Ts))

        # calculate the number of range and Doppler Cells
        Nr = int(ceil(2*self.R_max/delta_r))
        Nv = int(ceil(2*PulseRadar.Fs/delta_v))

        # generate range and Doppler FFT grids
        r_grid = np.linspace(0, Nr*delta_r, Nr, endpoint=False)
        v_grid = np.linspace(-Nv/2, Nv/2-1, Nv)*delta_v

        #generate range-Doppler map
        self.doppler_spectrum = np.zeros((self.total_samples, 127), dtype='complex_')
        for i in range(self.total_samples):
            self.doppler_spectrum[i,:] = np.fft.fftshift(np.fft.fft(self.pri_data[i,:], 127))

        # generate velocity grid
        lambda_c = PulseRadar.sound_speed/9500
        v_max = lambda_c/(4*((self.pw_samples+self.lstn_samples)*PulseRadar.Ts))
        v_grid = np.linspace(-v_max, v_max, Nv)

        # normalise the results of the range doppler plot
        max_value = np.max(np.abs(self.doppler_spectrum))
        self.doppler_spectrum1 = np.abs(self.doppler_spectrum)/max_value

        doppler = np.log10(np.abs(self.doppler_spectrum.T))
        
        self.extent = [r_grid[0], r_grid[-1]/2, v_grid[-1], v_grid[0]]
        #plt.imshow(np.abs(self.doppler_spectrum.T), aspect='auto', cmap='jet', extent=self.extent)
        #plt.xlabel('Range (m)')
        #plt.ylabel('Velocity (m/s)')
        #plt.title('Range-Doppler map')
        #plt.colorbar()
        #plt.show()

    def cfar_detection(self):
        cfar_size = (7, 5)

        # number of guard and training cels
        num_guards = 2
        num_training = 4

        range_bins = self.total_samples
        doppler_bins = 127

        # perform the cfar detection
        #cfar_detect = np.abs(convolve2d(self.doppler_spectrum1, cfar_mask, 'same'))
        data = np.abs(self.doppler_spectrum)
        cfar_detect = np.copy(data)
        
        # Iterate over the range and doppler bins
        for i in range(num_guards+2, range_bins - num_guards):
            #print(i)
            for j in range(num_guards+2, doppler_bins - num_guards):
                # Calculate the local noise level using the guard cells
                noise_level = np.mean(data[i-num_guards:i+num_guards+1, j-num_guards:j+num_guards+1])
                # Calculate the threshold using the training cells
                threshold = 0.35+np.mean(data[i-num_training:i+num_training+1, j-num_training:j+num_training+1])
                # Check if the cell value exceeds the threshold
                if data[i, j] > threshold:
                    cfar_detect[i, j] = 10
                else:
                    cfar_detect[i, j] = 0

        #plt.imshow(cfar_detect.T, aspect='auto', cmap='gray', extent=self.extent)
        #plt.xlabel('Range (m)')
        #plt.ylabel('Velocity (m/s)')
        #plt.title('CFAR result')
        #plt.colorbar()
        #plt.show()
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        im0 = ax[0].imshow(np.abs(self.doppler_spectrum.T), aspect='auto', cmap='jet', extent=self.extent)
        ax[0].set_xlabel('Range (m)')
        ax[0].set_ylabel('Velocity (m/s)')
        #ax[0].set_title('Range-Doppler map')
        im1 = ax[1].imshow(cfar_detect.T, aspect='auto', cmap="jet", extent=self.extent)
        ax[1].set_xlabel('Range (m)')
        ax[1].set_ylabel('Velocity (m/s)')
        #ax[1].set_title('CFAR result')
        fig.colorbar(im0)
        fig.colorbar(im1)
        plt.show()

def main():
    instance = PulseRadar(pulses=16, start_freq=8e3, max_dist=7, min_dist=3, bandwidth=3e3)
    instance.generate_signal()

    # defines the threads for playing and recording
    ithread = Thread(target=instance.listener)
    othread = Thread(target=instance.emitter)
    # starts the playing and recording threads
    ithread.start()
    othread.start()

    print( "Waiting for the playing and recording threads to finish..." )
    time.sleep(PulseRadar.duration + 1)

    #instance.savefile();
    instance.terminate_pa()
    instance.pulse_compression()
    #instance.data_matrix()
    #instance.doppler_processing()
    #instance.cfar_detection()

if __name__ == '__main__':
    main()

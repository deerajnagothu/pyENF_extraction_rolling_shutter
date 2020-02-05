
addpath('Recordings');

m = VideoReader('MVI_0292.MP4');

numberOfFrames = round(m.Duration * m.FrameRate);

row_signal = zeros(1, numberOfFrames * m.Height);
index = 1;
while hasFrame(m)
    vidFrame = readFrame(m);
    meanAcrossRGB = mean(vidFrame,3);
    meanAcrossColumns = mean(meanAcrossRGB,2);
    row_signal(index:index+m.Height - 1) = meanAcrossColumns';
    index = index + m.Height;
end

% approximately 21578 samples per second 

frame_size = 2000;  % in seconds
fs = round(m.FrameRate * m.Height);
nfft = 2048;
overlap_amount = 0;
spectrogram(row_signal, round(hamming(frame_size)), round(overlap_amount), nfft, fs);

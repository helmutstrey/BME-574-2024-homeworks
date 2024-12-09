import Pkg
Pkg.activate(".")
##
Pkg.add(["FFTW","Random","LinearAlgebra","GLMakie","Distributions"])

using FFTW, Random, LinearAlgebra, GLMakie, Distributions


#based on Example from Brunton and Kutz
## Generate signal, DCT of signal
n = 4096; # points in high resolution signal
t = range(0, 1, n+1);# [I'm improving the sampling to actually be 1/n rather than 1/(n-1)]
t = t[2:end];
x = cos.(2* 97 * pi .* t) + cos.(2* 777 * pi .* t);
xfd = fft(x); # Fourier transformed signal
PSD = abs.(xfd.*conj(xfd))/n; # Power spectral density

#[Generate freq. ax.]
dt =t[1];
Nyq = 1/(2*dt);
f = ( 0 : ((n-1)/2) ) / (n*dt);
if mod(n,2) == 1
    #odd, no Nyq component
    f2side = [f;-f[end:-1:2]];#odd n
    xfd[1,end] = xfd[1,1] / 2;
else
    #even, has Nyq component
    f2side = [f;-Nyq;-f[end:-1:2]];#even n
    f = [f;Nyq];
    xfd[1,end] = xfd[1,end] / 2;
end

f1sz = length(f);


##Randomly sample signal
nf = 0.01;
p = 128; # num. random samples, p=n/32
perm = rand(1:n,p);
y = x[perm]; # compressed measurement
if nf != 0
    nd = Normal(0,0.1)
    y= y.+rand(nd,size(perm));
end
##


## [Measure] compressed sensing problem
Ψ = dct(1.0 * I(n), 1); # build Ψ from discrete cosine transform
Θ = Ψ[perm, :]; # Measure rows of Ψ
# s = cosamp(Θ,y',10,1.e-10,10); # CS via matching pursuit
# xrecon = idct(s); # reconstruct full signal



## Mine from here on:
function reconAndFT(s,n)
    xr = idct(s,1);#Reconstruct
    xr_FD = fft(xr); #Fourier Transform
    xr_PSD = abs.(xr_FD.*conj.(xr_FD))/n; #PSD
    return (xr,xr_FD,xr_PSD)
end
## Reconstruct Signal
# Least Squares
s_LS = Θ\(y);
xLS,xS_FD,xLS_PSD = reconAndFT(s_LS,n);

# Orthogonal Matching Pursuit
println("OMP")
include("OMP.jl")
s_OMP,r_OMP,normR_OMP,residHist_OMP, errHist_OMP = OMP(Θ,y,10);
xOMP,xOMP_FD,xOMP_PSD = reconAndFT(s_OMP,n);
println("")

# Compressive Sampling Matching Pursuit (CoSaMP)
println("CoSaMP")
include("CoSaMP.jl")
s_CoSaMP,r_CoSaMP,normR_CoSaMP,residHist_CoSaMP, errHist_CoSaMP = CoSaMP(Θ,y,10);
xCoSaMP,xCoSaMP_FD,xCoSaMP_PSD = reconAndFT(s_CoSaMP,n);
println("")


## Figures
f0 = Figure()
tdax0 = Axis(f0[1,1],
    xlabel = "Time [s]",
    ylabel = "Signal"
)
fdax0 = Axis(f0[1,2],
    xlabel = "Frequency [Hz]",
    ylabel = "Signal PSD"
)
tdaxRec = Axis(f0[2,1],
    xlabel = "Time [s]",
    ylabel = "Signal"
)
fdaxRec = Axis(f0[2,2],
    xlabel = "Frequency [Hz]",
    ylabel = "Signal PSD"
)

# original data
#Time domain
lines!(tdax0,t,x,
    linewidth = 2, label = "Original Signal", color = :black
)
scatter!(tdax0,t[perm],y,
    marker = :star5, color = :red, markersize = 15, label = "Sample Points"
)
#=
lines!(tdaxRec,t,x,
    linewidth = 2, label = "Original Signal", linestyle = :dash, color = :black
)
=#

scatter!(tdaxRec,t[perm],y,
    marker = :star5, color = :red, markersize = 15, label = "Sample Points"
)

#Frequency Domain
lines!(fdax0,f,PSD[1:f1sz],
    linewidth = 2, label = "Original Signal", color = :black
)
lines!(fdaxRec,f,PSD[1:f1sz],
    linewidth = 2, label = "Original Signal", linestyle = :dash, color = :black
)

scatter!(fdax0,1,NaN,
    marker = :star5, color = :red, markersize = 15, label = "Sample Points"
)
xlims!(tdax0,0.25,0.35)
xlims!(tdaxRec,0.25,0.35)
xlims!(fdax0,0,f[end])
xlims!(fdaxRec,0,f[end])
linkxaxes!(tdax0,tdaxRec)
linkxaxes!(fdax0,fdaxRec);
lg0   = axislegend(fdax0,position = :rt)

#lg0   = axislegend(fdax0,position = :rt)
#lgRec = axislegend(fdaxRec,position = :rt)
f0

## Reconstructions

## Least-Squares
#Time Domain
#=
lines!(tdaxRec,t,xLS,
linewidth = 2, label = "Least-Squares" 
)
#Frequency Domain
lines!(fdaxRec,f,xLS_PSD[1:f1sz],
linewidth = 2, label = "Least-Squares", 
)

=#

## OMP
#Time Domain
lines!(tdaxRec,t,xOMP,
    linewidth = 2, label = "Orthogonal Matching Pursuit (OMP)" 
)
#Frequency Domain
lines!(fdaxRec,f,xOMP_PSD[1:f1sz],
    linewidth = 2, label = "Orthogonal Matching Pursuit (OMP)", 
)



## CoSaMP
#Time Domain
lines!(tdaxRec,t,xCoSaMP,
    linewidth = 2, label = "Compressive Sampling Matching Pursuit (CoSaMP)" 
)
#Frequency Domain
lines!(fdaxRec,f,xCoSaMP_PSD[1:f1sz],
    linewidth = 2, label = "Compressive Sampling Matching Pursuit (CoSaMP)", 
)




lgRec = axislegend(fdaxRec,position = :rt)
f0
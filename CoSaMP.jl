function CoSaMP(A, b, k=size(b,1), errFcn = [], opts=[])
    
    # --Parse inupts--
    # For now I'm just manually predefining the opts versions
    printEvery  = 1
    maxiter     = 1000
    normTol     = 1e-10
    cg_tol      = 13-6
    cg_maxit    = 20
    HSS         = false
    TWO_SOLVES  = false
    addK        = 2*k 
    support_tol = 1e-10
    
    #(here I'm implying that the matlab LARGESCALE is false, i.e., that A is a matrix rather than a cell array of fucntions defining Af(x) and At(x)
    Af(X) = A*x
    At(x) = A'*x

     # --Initialize--
    r     = b          # residual (initialized to equal signal) 
    Ar    = At(r)      # vector of projections from the residula to the sample matrix
    N     = size(Ar,1) # number of atoms
    M     = size(r,1)  # size of atoms
    if k > M/3
        error("k cannot be larger than the dimension of the atoms")
    end
    x           = zeros(N,1)    # approximated vector
    ind_k       = [];

    indx_set    = zeros(k,1)    # set of indices of A?
    idx_set_sorted = zeros(k,1) # sorted set of indices (unused?)
    A_T         = zeros(M,k)    # T components of A
    A_T_nonorth = zeros(M,k)    # non-orthogonal
    residHist   = zeros(maxiter,1)
    errHist     = zeros(maxiter,1)
    normR = norm(r)
    kkout = 0 #"outer" keyword made kk a global variable

    print("Iter,   |T|,  Resid")
    if !isempty(errFcn)
        print(",   Error")
    end
    print("\n")
    for kk = 1:maxiter
        #--Step 1: find new index and atom to add
        y_sort = sort(abs.(Ar), rev = true)
        cutoff = y_sort[addK]
        cutoff = max(cutoff,support_tol)
        ind_new = findall(abs.(Ar) .>= cutoff)

        # -- Merge
        T = union(ind_new,ind_k)
        
        # -- Step 2: update residual
        if HSS
            RHS         = r;# where r = b - A*x, so we'll need to add in "x" later
            x_warmstart = zeros(length(T));
        else
            RHS         = b;
            x_warmstart = x[T];
        end

        #-- solve for x on the suppor set "T"
        #[Ignoring LARGESCALE option]
        x_T = A[:,T]\RHS;   # more efficient; equivalent to pinv when underdetermined.
        #x_T     = pinv( A(:,T) )*RHS;

        if HSS
            # HSS variation of CoSaMP
            x_new   = zeros(N)
            x_new[T] = x_T
            x       = x+x_new #this is the key extra step in HSS
            cutoff  = findCutoff(x, k)
            x       = x.* (abs.(x) > cutoff)
            ind_k   = findall(x)
            
            if TWO_SOLVES
                #[Ignoring LARGESCALE option]
                x_T2  = A[:,ind_k]\b
                x[ ind_k ] = x_T2
            end

            #update r
            r_old = r
            r = b - Af(x)
        else
            #Standard CoSaMP
            #Note: note this is implemented "slightly" more efficiently than the HSS version

            # Prune x to keep only "k" entries
            cutoff = findCutoff(x_T, k)
            Tk = findall(abs.(x_T) .>= cutoff)
            #This is assuming there are no ties. If there are, from a practical standpoint
            # it probably doesn't matter much what you do. So out of laziness, we don't worry
            # about it.
            ind_k = T[Tk]
            x = 0 .* x
            x[ ind_k ] = x_T[Tk]

            if TWO_SOLVES
                #[Ignoring LARGESCALE option]
                x_T2 = A[:,indk]\b
                x[ ind_k ] = x_T2
            end
            r_old = r
            #[Ignoring LARGESCALE option]
            r = b - A[:,ind_k]*x_T[Tk]
        end


        # -- Print some info --
        printnow = (mod(kk,printEvery) == 0 || kk == maxiter)
        normR = norm(r)
        STOP = false
        if normR < normTol || norm(r-r_old) < normTol
            STOP = true
            printnow = true
        end
        
        
        if !isempty(errFcn) 
            er = errFcn(x)
            errHist[kk] = er
        end
        if printnow
            #print
            print("$(kk), $(length(T)), $(normR)")
            if !isempty(errFcn)
                print(", $(er)")
                #[Ignoring LARGESCALE option]
            end
            println("")#print newline
        end
        residHist[kk] = normR

        # check for halt condition or prepare next loop
        if STOP
            println("Reached stopping criteria")
            kkout = kk
            break
        end
        
        if kk < maxiter
            Ar = At(r)
        else
            kkout = kk
        end
    end
    return (vec(x), r, normR, residHist[1:kkout], errHist[1:kkout]) 
end

function findCutoff(x,k)
    #finds the appropriate cutoff such that after hard-threshoding,
    #"x" will be k-sparse
    x = sort(abs.(x),rev = true)
    if k > length(x)
        tau = x[end]*0.999;
    else
        tau = x[k];
    end
    return tau
end
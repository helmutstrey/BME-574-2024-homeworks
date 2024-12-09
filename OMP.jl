function OMP(A, b, k=size(b,1), errFcn = [], opts=[])
    # --Parse inupts--
    # For now I'm just manually predefining the opts versions
    slowmode = false
    printEvery = 1
    target_resid = -Inf # in the matlab version, if k is not an integer (or is a cell) it represents this and k = size(b,1)

    #[here I'm implying that the matlab LARGESCALE is false, i.e., that A is a matrix rather than a cell array of  fucntions defining Af(x) and At(x)]
    Af(s) = A*s
    At(s) = A'*s
    
    # --Initialize--
    r     = b          # residual (initialized to equal signal) 
    normR = norm(r)    # norm of residual
    Ar    = At(r)      # vector of projections from the residula to the sample matrix
    N     = size(Ar,1) # number of atoms
    M     = size(r,1)  # size of atoms
    if k > M
        error("k cannot be larger than the dimension of the atoms")
    end
    unitVector  = zeros(N,1)    # this just preallocates a vector which is set to be the unit vector along a given index when LARGESCALE = true
    x           = zeros(N,1)    # approximated vector

    indx_set    = zeros(Int64,k)    # set of indices of A?
    idx_set_sorted = zeros(Int64,k) # sorted set of indices (unused?)
    A_T         = zeros(M,k)    # T components of A
    A_T_nonorth = zeros(M,k)    # non-orthogonal
    residHist   = zeros(k,1)
    errHist     = zeros(k,1)
    kkout = 0 #"outer" keyword made kk a global variable

    for kk = 1:k
        #--Step 1: find new index and atom to add
        ind_new = argmax(abs.(Ar))
        # Matlab code contains a (commented) check if this index is already in the index set

        
        indx_set[kk] = ind_new
        #indx_set_sorted[1:kk] = sort(indx_set[1:kk]) # appears to be unused, also not clear why it's here rather than after the loop
        
        # [if LARGESCALE -> evaluate differently, not used here yet]
        atom_new    = A[:,ind_new];
        A_T_nonorth[:,kk] = atom_new# "before orthogonalizing and such"

        #--Step 2: update residual
        if slowmode
            x_T = A_T_nonorth[:,1:kk]\b
            # commented code here that more explicitly uses QR decomposition
            
            x[indx_set[1:kk]] = x_T
            r = b - A_T_nonorth[:,1:kk]*x_T
        else
            # "First orthogonalize 'atom_new' against all previous atoms"
            # "We use MGS [modified Gram-Schmidt]"
            for j = 1:(kk-1)
                atom_new = atom_new - (A_T[:,j]' * atom_new)*A_T[:,j]
            end
            # Second, normalize:
            atom_new  = atom_new./norm(atom_new)
            A_T[:,kk] = atom_new

            # "Third, solve least-squares problem (which is now very easy since A_T[:,1:k] is orthogonal)"
            x_T = A_T[:,1:kk]' * b

            
            x[indx_set[1:kk]] .= x_T # "note: indx_set is guaranteed to never shrink

            # Fourth, update residual:
            # " r = b - Af(x); # wrong!"
            r = b - A_T[:,1:kk]*x_T

            # "N.B. This err is unreliable since this "x" is not the same 
            #  (since it relies on A_T, which is the orthogonalized version)."            
        end

        normR = norm(r)

        # -- Print some info --
        printnow = (mod(kk,printEvery) == 0 || kk == k)
        if 0 < printEvery < Inf && (normR < target_resid)
            # this is the final iteration so print no matter what
            printnow = true
        end
        
        if !isempty(errFcn) && slowmode
            er = errFcn(x)
            if printnow
                #print
                println("$(kk), $(normR), $(er)")
            end
        else
            if printnow
                #print
                println("$(kk), $(normR)")
                # "(if this is not ins slowmode, the error is unreliable)"
            end
        end
        residHist[kk] = normR

        # check for halt condition or prepare next loop
        if normR < target_resid
            if printnow
                #print " Residual reached...
                println("Residual reached dsired size ($(normR) < $(target_resid))")
            end
            kkout = kk#keep number of repetitions available
            break
        elseif kk < k
            Ar = At(r)
        else
            #print "number of repetitions reached without reaching target residual"
            println("Maximum repetitions reached without reaching target residual ($(normR) > $(target_resid))")
            kkout = kk#keep number of repetitions available
        end
        
    end
    if !slowmode # "(ins slowMode, we already have this info)
        # "For the last iteration, we need to do this without orthoganalizing A so that the x coefficients match what is exected"
        x_T = A_T_nonorth[:,1:kkout]\b
        x[indx_set[1:kkout]] = x_T
    end
    r = b - A_T_nonorth[:,1:kkout] * x_T


    normR = norm(r)
    return (vec(x), r, normR, residHist, errHist) 
end
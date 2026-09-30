/* Constraints and packed Jacobian for a multilayer tanh NNOE.
 *
 * [h,J] = nnoe_constraints_mex(theta,y,u,n,Nn)
 * Variable ordering is [theta; vec(y')], matching sidqr_mex with q=0.
 */
#include "mex.h"
#include <math.h>
#include <string.h>

static int scalar_int(const mxArray *a,const char *name)
{
    double x;
    if(!mxIsDouble(a)||mxIsComplex(a)||mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("nnoe_constraints:arg","%s must be a real scalar.",name);
    x=mxGetScalar(a);
    if(x<0||x!=(int)x) mexErrMsgIdAndTxt("nnoe_constraints:arg","%s must be a nonnegative integer.",name);
    return (int)x;
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    int N,n,p,q,L,in0,nth=0,maxH=0,sumH=0;
    int *H,*woff,*boff,*aoff;
    int ell,k,r,i,j,s,layer,c,beta;
    const double *theta,*y,*u,*Nv;
    double *h,*Th,*Bd,*act,*der,*delta,*prev,*input;
    mxArray *ThArray=NULL,*BdArray=NULL,*packed=NULL;
    mwSize thetaCount;

    if(nrhs!=5||nlhs>2) mexErrMsgIdAndTxt("nnoe_constraints:usage",
        "[h,J]=nnoe_constraints_mex(theta,y,u,n,Nn)");
    for(i=0;i<3;++i) if(!mxIsDouble(prhs[i])||mxIsComplex(prhs[i]))
        mexErrMsgIdAndTxt("nnoe_constraints:type","theta, y and u must be real double arrays.");
    n=scalar_int(prhs[3],"n"); N=(int)mxGetM(prhs[1]); p=(int)mxGetN(prhs[1]);
    q=(int)mxGetN(prhs[2]);
    if((int)mxGetM(prhs[2])!=N||N<=n||p<1)
        mexErrMsgIdAndTxt("nnoe_constraints:dims","y and u must have N rows and N must exceed n.");
    if(!mxIsDouble(prhs[4])||mxIsComplex(prhs[4]))
        mexErrMsgIdAndTxt("nnoe_constraints:type","Nn must be a real double vector.");
    L=(int)mxGetNumberOfElements(prhs[4]); if(L<1) mexErrMsgIdAndTxt("nnoe_constraints:dims","Nn cannot be empty.");
    Nv=mxGetDoubles(prhs[4]);
    H=(int*)mxMalloc(L*sizeof(int)); woff=(int*)mxMalloc((L+1)*sizeof(int));
    boff=(int*)mxMalloc((L+1)*sizeof(int)); aoff=(int*)mxMalloc(L*sizeof(int));
    in0=n*p+(n+1)*q;
    for(layer=0;layer<L;++layer){H[layer]=(int)Nv[layer];if(H[layer]<1||Nv[layer]!=H[layer])mexErrMsgIdAndTxt("nnoe_constraints:dims","Nn entries must be positive integers.");if(H[layer]>maxH)maxH=H[layer];aoff[layer]=sumH;sumH+=H[layer];}
    for(layer=0;layer<L;++layer){int nin=(layer==0?in0:H[layer-1]);woff[layer]=nth;nth+=H[layer]*nin;boff[layer]=nth;nth+=H[layer];}
    woff[L]=nth; nth+=p*H[L-1]; boff[L]=nth; nth+=p;
    thetaCount=mxGetNumberOfElements(prhs[0]);
    if(thetaCount!=(mwSize)nth) mexErrMsgIdAndTxt("nnoe_constraints:theta","theta has %llu elements; expected %d.",(unsigned long long)thetaCount,nth);
    theta=mxGetDoubles(prhs[0]); y=mxGetDoubles(prhs[1]); u=mxGetDoubles(prhs[2]);
    beta=(n+1)*p; c=p*(N-n);
    plhs[0]=mxCreateDoubleMatrix(c,1,mxREAL); h=mxGetDoubles(plhs[0]);
    if(nlhs>1){
        const char *fields[]={"N","n","p","q","ntheta","parameterBlock","bandBlock"};
        packed=mxCreateStructMatrix(1,1,7,fields);
        ThArray=mxCreateDoubleMatrix(nth,c,mxREAL); BdArray=mxCreateDoubleMatrix(beta,c,mxREAL);
        Th=mxGetDoubles(ThArray); Bd=mxGetDoubles(BdArray);
        mxSetField(packed,0,"N",mxCreateDoubleScalar(N)); mxSetField(packed,0,"n",mxCreateDoubleScalar(n));
        mxSetField(packed,0,"p",mxCreateDoubleScalar(p)); mxSetField(packed,0,"q",mxCreateDoubleScalar(0));
        mxSetField(packed,0,"ntheta",mxCreateDoubleScalar(nth));
        mxSetField(packed,0,"parameterBlock",ThArray); mxSetField(packed,0,"bandBlock",BdArray);
        plhs[1]=packed;
    }else {Th=NULL;Bd=NULL;}
    act=(double*)mxMalloc(sumH*sizeof(double)); der=(double*)mxMalloc(sumH*sizeof(double));
    delta=(double*)mxMalloc(maxH*sizeof(double)); prev=(double*)mxMalloc(maxH*sizeof(double));
    input=(double*)mxMalloc((in0?in0:1)*sizeof(double));

    for(ell=0;ell<N-n;++ell){
        k=n+ell;
        s=0;
        for(i=1;i<=n;++i)for(j=0;j<p;++j)input[s++]=y[(k-i)+(mwSize)N*j];
        for(i=0;i<=n;++i)for(j=0;j<q;++j)input[s++]=u[(k-i)+(mwSize)N*j];
        for(layer=0;layer<L;++layer){
            int nin=(layer==0?in0:H[layer-1]); const double *xin=(layer==0?input:act+aoff[layer-1]);
            for(i=0;i<H[layer];++i){
                double z=theta[boff[layer]+i];
                if(layer==0){
                    int ny=n*p,nu=(n+1)*q;
                    for(j=0;j<ny;++j)z+=theta[woff[0]+i*ny+j]*xin[j];
                    for(j=0;j<nu;++j)z+=theta[woff[0]+H[0]*ny+i*nu+j]*xin[ny+j];
                }else for(j=0;j<nin;++j)z+=theta[woff[layer]+i*nin+j]*xin[j];
                act[aoff[layer]+i]=tanh(z);der[aoff[layer]+i]=1.0-act[aoff[layer]+i]*act[aoff[layer]+i];
            }
        }
        for(r=0;r<p;++r){
            int col=ell*p+r; double out=theta[boff[L]+r];
            for(i=0;i<H[L-1];++i)out+=theta[woff[L]+r*H[L-1]+i]*act[aoff[L-1]+i];
            h[col]=out-y[k+(mwSize)N*r];
            if(!Th&&!Bd)continue;
            if(Th){for(i=0;i<H[L-1];++i)Th[woff[L]+r*H[L-1]+i+(mwSize)nth*col]=act[aoff[L-1]+i];Th[boff[L]+r+(mwSize)nth*col]=1.0;}
            for(i=0;i<H[L-1];++i)delta[i]=theta[woff[L]+r*H[L-1]+i]*der[aoff[L-1]+i];
            for(layer=L-1;layer>=0;--layer){
                int nin=(layer==0?in0:H[layer-1]); const double *xin=(layer==0?input:act+aoff[layer-1]);
                if(Th)for(i=0;i<H[layer];++i){
                    if(layer==0){
                        int ny=n*p,nu=(n+1)*q;
                        for(j=0;j<ny;++j)Th[woff[0]+i*ny+j+(mwSize)nth*col]=delta[i]*xin[j];
                        for(j=0;j<nu;++j)Th[woff[0]+H[0]*ny+i*nu+j+(mwSize)nth*col]=delta[i]*xin[ny+j];
                    }else for(j=0;j<nin;++j)Th[woff[layer]+i*nin+j+(mwSize)nth*col]=delta[i]*xin[j];
                    Th[boff[layer]+i+(mwSize)nth*col]=delta[i];
                }
                if(layer==0){
                    if(Bd){
                        for(i=1;i<=n;++i)for(j=0;j<p;++j){double z=0.0;int ix=(i-1)*p+j;for(s=0;s<H[0];++s)z+=theta[woff[0]+s*(n*p)+ix]*delta[s];Bd[(n-i)*p+j+(mwSize)beta*col]=z;}
                        Bd[n*p+r+(mwSize)beta*col]=-1.0;
                    }
                }else{
                    for(j=0;j<H[layer-1];++j){double z=0.0;for(i=0;i<H[layer];++i)z+=theta[woff[layer]+i*H[layer-1]+j]*delta[i];prev[j]=z*der[aoff[layer-1]+j];}
                    memcpy(delta,prev,H[layer-1]*sizeof(double));
                }
            }
        }
    }
    mxFree(input);mxFree(prev);mxFree(delta);mxFree(der);mxFree(act);
    mxFree(aoff);mxFree(boff);mxFree(woff);mxFree(H);
}

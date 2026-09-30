#include "mex.h"

static double scalar(const mxArray *a,const char *name){
    if(!mxIsDouble(a)||mxIsComplex(a)||mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("levitator_constraints:arg","%s must be a real scalar.",name);
    return mxGetScalar(a);
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
    mwSize N,i,rows; const double *x,*u; double Ts,m,g,*h,*Th,*Bd,km,k0,c,*y;
    mxArray *packed=NULL,*ThArray=NULL,*BdArray=NULL;
    if(nrhs!=5||nlhs>2)mexErrMsgIdAndTxt("levitator_constraints:usage","[h,J]=levitator_constraints_mex(x,u,Ts,m,g)");
    if(!mxIsDouble(prhs[0])||!mxIsDouble(prhs[1])||mxIsComplex(prhs[0])||mxIsComplex(prhs[1]))
        mexErrMsgIdAndTxt("levitator_constraints:type","x and u must be real double vectors.");
    if(mxGetNumberOfElements(prhs[0])<6)
        mexErrMsgIdAndTxt("levitator_constraints:size","x must contain three parameters and at least three outputs.");
    N=mxGetNumberOfElements(prhs[0])-3;
    if(mxGetNumberOfElements(prhs[1])<N)mexErrMsgIdAndTxt("levitator_constraints:size","Invalid x or u length.");
    x=mxGetDoubles(prhs[0]);u=mxGetDoubles(prhs[1]);Ts=scalar(prhs[2],"Ts");m=scalar(prhs[3],"m");g=scalar(prhs[4],"g");
    if(Ts<=0)mexErrMsgIdAndTxt("levitator_constraints:Ts","Ts must be positive.");
    km=x[0];k0=x[1];c=x[2];y=(double*)(x+3);rows=N-2;
    plhs[0]=mxCreateDoubleMatrix(rows,1,mxREAL);h=mxGetDoubles(plhs[0]);
    if(nlhs>1){
        const char *fields[]={"N","n","p","q","ntheta","parameterBlock","bandBlock"};
        packed=mxCreateStructMatrix(1,1,7,fields);
        ThArray=mxCreateDoubleMatrix(3,rows,mxREAL);BdArray=mxCreateDoubleMatrix(3,rows,mxREAL);
        Th=mxGetDoubles(ThArray);Bd=mxGetDoubles(BdArray);
        mxSetField(packed,0,"N",mxCreateDoubleScalar((double)N));mxSetField(packed,0,"n",mxCreateDoubleScalar(2));
        mxSetField(packed,0,"p",mxCreateDoubleScalar(1));mxSetField(packed,0,"q",mxCreateDoubleScalar(0));
        mxSetField(packed,0,"ntheta",mxCreateDoubleScalar(3));
        mxSetField(packed,0,"parameterBlock",ThArray);mxSetField(packed,0,"bandBlock",BdArray);
        plhs[1]=packed;
    }else {Th=NULL;Bd=NULL;}
    for(i=0;i<rows;++i){
        double y0=y[i],y1=y[i+1],y2=y[i+2];
        double acceleration=(y2-2*y1+y0)/(Ts*Ts);
        double velocity=(y1-y0)/Ts;
        double ySquared=y0*y0;
        h[i]=m*ySquared*(acceleration-g)+km*u[i]*u[i]+k0+c*ySquared*velocity;
        if(Th){
            Th[0+3*i]=u[i]*u[i];Th[1+3*i]=1.0;Th[2+3*i]=ySquared*velocity;
            Bd[0+3*i]=2*m*y0*(acceleration-g)+m*ySquared/(Ts*Ts)
                +c*(2*y0*velocity-ySquared/Ts);
            Bd[1+3*i]=-2*m*ySquared/(Ts*Ts)+c*ySquared/Ts;
            Bd[2+3*i]=m*ySquared/(Ts*Ts);
        }
    }
}

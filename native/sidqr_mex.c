/* Native linear-algebra step for FL-CMO with a packed structured Jacobian. */
#include "mex.h"
#include "sidqr.h"
#include <string.h>

static const mxArray *field(const mxArray *s,const char *name)
{
    const mxArray *value=mxGetField(s,0,name);
    if(!value)mexErrMsgIdAndTxt("sidqr:field","Missing Jacobian field '%s'.",name);
    return value;
}
static int integer_field(const mxArray *s,const char *name)
{
    const mxArray *a=field(s,name);double value;
    if(!mxIsDouble(a)||mxIsComplex(a)||mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("sidqr:field","Field '%s' must be a scalar.",name);
    value=mxGetScalar(a);
    if(value<0||value!=(int)value)
        mexErrMsgIdAndTxt("sidqr:field","Field '%s' must be a nonnegative integer.",name);
    return (int)value;
}
static mxArray *stats_output(const sidqr_stats *stats,int blockSize)
{
    const char *names[]={"factorizationTime","solveTime","flops","peakFrontHeight",
        "workspaceBytes","factorBytes","inputBytes","blockSize","backend"};
    mxArray *value=mxCreateStructMatrix(1,1,9,names);
    mxSetField(value,0,"factorizationTime",mxCreateDoubleScalar(stats->t_fact));
    mxSetField(value,0,"solveTime",mxCreateDoubleScalar(stats->t_solve));
    mxSetField(value,0,"flops",mxCreateDoubleScalar(stats->flops));
    mxSetField(value,0,"peakFrontHeight",mxCreateDoubleScalar(stats->Hmax));
    mxSetField(value,0,"workspaceBytes",mxCreateDoubleScalar((double)stats->strip_peak));
    mxSetField(value,0,"factorBytes",mxCreateDoubleScalar((double)stats->R_bytes));
    mxSetField(value,0,"inputBytes",mxCreateDoubleScalar((double)stats->input_bytes));
    mxSetField(value,0,"blockSize",mxCreateDoubleScalar(blockSize));
    mxSetField(value,0,"backend",mxCreateString("packed Q-less Householder projection"));
    return value;
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    sidqr_dims d;sidqr_stats stats;int N,n,p,q,nth,blockSize,column,row,top,status;
    const mxArray *parameterArray,*bandArray;
    const double *parameterBlock,*bandBlock,*gradient,*constraints;
    double gain,*rhs,*multipliers,*velocity;
    if(nrhs!=4||nlhs>2||!mxIsStruct(prhs[0]))
        mexErrMsgIdAndTxt("sidqr:usage","[velocity,stats]=sidqr_mex(J,gradient,constraints,gain)");
    N=integer_field(prhs[0],"N");n=integer_field(prhs[0],"n");p=integer_field(prhs[0],"p");
    q=integer_field(prhs[0],"q");nth=integer_field(prhs[0],"ntheta");
    if(sidqr_setdims(&d,N,n,p,q,nth))mexErrMsgIdAndTxt("sidqr:dims","Invalid Jacobian dimensions.");
    parameterArray=field(prhs[0],"parameterBlock");bandArray=field(prhs[0],"bandBlock");
    if(!mxIsDouble(parameterArray)||!mxIsDouble(bandArray)||!mxIsDouble(prhs[1])||
       !mxIsDouble(prhs[2])||!mxIsDouble(prhs[3])||mxIsComplex(parameterArray)||
       mxIsComplex(bandArray)||mxIsComplex(prhs[1])||mxIsComplex(prhs[2])||
       mxIsComplex(prhs[3])||mxGetM(parameterArray)!=(mwSize)nth||
       mxGetN(parameterArray)!=(mwSize)d.m||mxGetM(bandArray)!=(mwSize)d.beta||
       mxGetN(bandArray)!=(mwSize)d.m||mxGetNumberOfElements(prhs[1])!=(mwSize)d.nu||
       mxGetNumberOfElements(prhs[2])!=(mwSize)d.m||mxGetNumberOfElements(prhs[3])!=1)
        mexErrMsgIdAndTxt("sidqr:size","Invalid packed blocks, gradient, constraints, or gain.");
    parameterBlock=mxGetDoubles(parameterArray);bandBlock=mxGetDoubles(bandArray);
    gradient=mxGetDoubles(prhs[1]);constraints=mxGetDoubles(prhs[2]);gain=mxGetScalar(prhs[3]);
    rhs=(double*)mxMalloc((size_t)d.m*sizeof(double));
    multipliers=(double*)mxMalloc((size_t)d.m*sizeof(double));
    if(!rhs||!multipliers)mexErrMsgIdAndTxt("sidqr:allocation","Workspace allocation failed.");

    /* Form -J*gradient + gain*h directly from the packed blocks. */
    for(column=0;column<d.m;++column){
        double product=0.0;top=nth+d.gamma*(column/p);
        for(row=0;row<nth;++row)product+=parameterBlock[row+(size_t)nth*column]*gradient[row];
        for(row=0;row<d.beta;++row)product+=bandBlock[row+(size_t)d.beta*column]*gradient[top+row];
        rhs[column]=-product+gain*constraints[column];
    }
    blockSize=12*p;memset(&stats,0,sizeof(stats));
    status=sidqr_solve_blas_lazy(&d,parameterBlock,bandBlock,blockSize,rhs,1,multipliers,1,&stats);
    if(status)mexErrMsgIdAndTxt("sidqr:failure","Structured QR solve failed (status %d).",status);

    /* Return -(gradient + J'*multipliers), again without expanding J. */
    plhs[0]=mxCreateDoubleMatrix(d.nu,1,mxREAL);velocity=mxGetDoubles(plhs[0]);
    for(row=0;row<d.nu;++row)velocity[row]=-gradient[row];
    for(column=0;column<d.m;++column){
        double scale=multipliers[column];top=nth+d.gamma*(column/p);
        for(row=0;row<nth;++row)velocity[row]-=parameterBlock[row+(size_t)nth*column]*scale;
        for(row=0;row<d.beta;++row)velocity[top+row]-=bandBlock[row+(size_t)d.beta*column]*scale;
    }
    mxFree(multipliers);mxFree(rhs);
    if(nlhs>1)plhs[1]=stats_output(&stats,blockSize);
}

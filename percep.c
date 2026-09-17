#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define epoca 30000
#define K 0.03f

float EntNt(float, float, float  );
float EntNtAND(float, float, float);
float EntNtOR(float, float, float);
float InitNt(float, float);
float sigmoide(float);
void pesos_initNt();


float Pesos[2];
float bias=0.5f;
float Error;

//Pesos de la neurona AND
float PesosAND[2];
float biasAND=0.5f;

//Pesos de la neurona OR
float PesosOR[2];
float biasOR=0.5f;
float EntNt( float x0, float x1, float target )
{

  float net = 0;
  float out = 0;
  float delta[2];  
  //float Error;

  //XOR salidas de AND y OR
  float targetAND = x0 * x1;
  float targetOR = (x0==1 || x1==1) ? 1.0f : 0.0f;
  float salidaAND = EntNtAND(x0, x1, targetAND);
  float salidaOR = EntNtOR(x0, x1, targetOR);

  net = Pesos[0]*salidaAND + Pesos[1]*salidaOR - bias;
  net = sigmoide( net );

  Error = target - net;
  //printf("Error for);
  bias -= K*Error;  
                    

  delta[0] = K*Error * salidaAND;
  delta[1] = K*Error * salidaOR; 


  Pesos[0] += delta[0];  
  Pesos[1] += delta[1]; 



  out=net;
  return out;
}


//entrenamiento de la neurona AND 
float EntNtAND( float x0, float x1, float target )
{
  float net = 0;
  float out = 0;
  float delta[2];

  net = PesosAND[0]*x0 + PesosAND[1]*x1 - biasAND;
  net = sigmoide( net );

  float ErrorAND = target - net;
  biasAND -= K*ErrorAND;

  delta[0] = K*ErrorAND * x0;
  delta[1] = K*ErrorAND * x1;

  PesosAND[0] += delta[0];
  PesosAND[1] += delta[1];

  out=net;
  return out;
}


//entrenamiento de la neurona OR
float EntNtOR( float x0, float x1, float target )
{
  float net = 0;
  float out = 0;
  float delta[2];

  net = PesosOR[0]*x0 + PesosOR[1]*x1 - biasOR;
  net = sigmoide( net );

  float ErrorOR = target - net;
  biasOR -= K*ErrorOR;

  delta[0] = K*ErrorOR * x0;
  delta[1] = K*ErrorOR * x1;

  PesosOR[0] += delta[0];
  PesosOR[1] += delta[1];

  out=net;
  return out;
}




float InitNt( float x0, float x1 )
{
  float net = 0;
  float out = 0;

net = 70.934807*x0 + 93.935219*x1 - 187.886169 ;

  net=sigmoide( net );

  out=net;
  return out;
}




void pesos_initNt(void)
{
int i;
  for(  i = 0; i < 2; i++ )
  {
    Pesos[i] = (float)rand()/RAND_MAX;

    // inicializar tambien los pesos de AND y OR
    PesosAND[i] = (float)rand()/RAND_MAX;
    PesosOR[i] = (float)rand()/RAND_MAX;
  }
}

float sigmoide( float s ){
  return (1/(1+ (-1*s)));
}


int main(){
  int i=0;
  float apr;
  pesos_initNt();

 while(i<epoca){

    printf("------------------------\n");
    printf("Salida Entrenamiento Epoca %d \n", i);
    apr=EntNt(1,1,0);
    printf("1,1=%f\n",apr);
    apr=EntNt(1,0,1);
    printf("1,0=%f\n",apr);
    apr=EntNt(0,1,1);
    printf("0,1=%f\n",apr);
    apr=EntNt(0,0,0);
    printf("0,0=%f\n",apr);
    printf("\n");
    printf("Pesos de cada epoca\n");
    printf("Peso 0 = %f\n", Pesos[0]);
    printf("Peso 1 = %f\n", Pesos[1]);

    printf("Bias = %f \n",bias);
	printf("Error %f\n ",Error  );
	printf("------------------------\n");
	i++;

}


  return 0;
}

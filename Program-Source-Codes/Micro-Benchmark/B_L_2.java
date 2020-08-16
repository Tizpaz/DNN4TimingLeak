import java.math.BigInteger;
import java.util.Random;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class B_L_2 {
    public static int N;
    public static int f0, f1, f2, f3, f4, f5, f6, f7;
    public static Random rand;
    public static void main(String[] args) throws InterruptedException {

        if (args.length > 0) {
          int secret = Integer.parseInt(args[0]);
          N = Integer.parseInt(args[1]);
          int counter = 0;
          long dur = 0;
          while(counter < 10) {
              long startTime = System.nanoTime();
              if (secret <= 10) {
                   f0();
              }
              else if (secret <= 15) {
                   f1();
              }
              else if (secret <= 20){
                   f2();
              }
              else if (secret <= 30){
                  f3();
              }
              else if (secret <= 35) {
                  f4();
              }
              else if (secret <= 40){
                  f5();
              }
              else if (secret <= 45){
                  f6();
              }
              else{
                  f7();
              }
              long endTime = System.nanoTime();
              long elapsedTime = endTime - startTime;
              double duration = (double)elapsedTime;
              dur = dur + (long) duration;
              counter++;
            }
            dur /= 10;
            FileWriter pw;
            try
            {
                File file = new File("B_L_2_time.csv");
                if (!file.exists()) {
                    file.createNewFile();
                }
                pw = new FileWriter(file.getAbsoluteFile(), true);
                pw.append(String.valueOf(N)+','+String.valueOf(f0)+','+String.valueOf(f1)+','+String.valueOf(f2)+','+String.valueOf(f3)+','+String.valueOf(f4)+','+String.valueOf(f5)+','+String.valueOf(f6)+','+String.valueOf(f7)+','+String.valueOf(dur));
                pw.append('\n');
                pw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    private static void f0() throws InterruptedException {
        f0 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(1);
        }
    }
    private static void f1() throws InterruptedException {
        f1 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(2);
        }
    }
    private static void f2() throws InterruptedException {
        f2 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(3);
        }
    }
    private static void f3() throws InterruptedException {
        f3 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(4);
        }
    }
    private static void f4() throws InterruptedException {
        f4 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(5);
        }
    }
    private static void f5() throws InterruptedException {
        f5 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(6);
        }
    }
    private static void f6() throws InterruptedException {
        f6 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(7);
        }
    }
    private static void f7() throws InterruptedException {
        f7 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(8);
        }
    }
}

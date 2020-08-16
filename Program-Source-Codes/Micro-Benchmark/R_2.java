import java.math.BigInteger;
import java.util.Random;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class R_2 {
    public static int N;
    public static int f0, f1;
    public static Random rand;
    public static void main(String[] args) throws InterruptedException {

        if (args.length > 0) {
          int secret = Integer.parseInt(args[0]);
          int secret0 = secret % 2;
          int secret1 = (secret/2) % 2;
          N = Integer.parseInt(args[1])/4;
          boolean isValid = true;
          int counter = 0;
          long dur = 0;
          while(counter < 10) {
              long startTime = System.nanoTime();
              if (secret0 == 0 && secret1 == 1) {
                for(int i = 0; i < 2 * N; i++) {
                    Thread.sleep(1);
                }
              }
              else if (secret0 == 1 && secret0 == 0) {
                for(int i = 0; i < N;i++) {
                    Thread.sleep(1);
                }
              }
              else{
                  isValid = true;
              }
              long endTime = System.nanoTime();
              long elapsedTime = endTime - startTime;
              double duration = (double)elapsedTime / 1000;
              dur = dur + (long) duration;
              counter++;
            }
            dur /= 10;
            FileWriter pw;
            try
            {
                File file = new File("R_2_time.csv");
                if (!file.exists()) {
                    file.createNewFile();
                }
                pw = new FileWriter(file.getAbsoluteFile(), true);
                pw.append(String.valueOf(N) + ',' + String.valueOf(f0) + ',' + String.valueOf(f1) + ',' +  String.valueOf(dur));
                pw.append('\n');
                pw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}

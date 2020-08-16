import java.math.BigInteger;
import java.util.Random;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class B_L_5 {
    public static int N;
    public static int f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60, f61, f62, f63;

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
              else if (secret <= 50){
                  f7();
              }
              else if (secret <= 60) {
                  f8();
              }
              else if (secret <= 65) {
                  f9();
              }
              else if (secret <= 70){
                  f10();
              }
              else if (secret <= 75){
                  f11();
              }
              else if (secret <= 80) {
                  f12();
              }
              else if (secret <= 85){
                  f13();
              }
              else if (secret <= 95){
                  f14();
              }
              else if (secret <= 105){
                  f15();
              }
              else if (secret <= 115) {
                  f16();
              }
              else if (secret <= 120){
                  f17();
              }
              else if (secret <= 125){
                  f18();
              }
              else if (secret <= 130) {
                  f19();
              }
              else if (secret <= 135){
                  f20();
              }
              else if (secret <= 140){
                  f21();
              }
              else if (secret <= 145){
                  f22();
              }
              else if (secret <= 150) {
                  f23();
              }
              else if (secret <= 155) {
                  f24();
              }
              else if (secret <= 160){
                  f25();
              }
              else if (secret <= 170){
                  f26();
              }
              else if (secret <= 175) {
                  f27();
              }
              else if (secret <= 185){
                  f28();
              }
              else if (secret <= 190){
                  f29();
              }
              else if (secret <= 195){
                  f30();
              }
              else if (secret <= 205) {
                  f31();
              }
              else if (secret <= 210) {
                  f32();
              }
              else if (secret <= 215) {
                  f33();
              }
              else if (secret <= 220){
                  f34();
              }
              else if (secret <= 225){
                  f35();
              }
              else if (secret <= 230) {
                  f36();
              }
              else if (secret <= 240){
                  f37();
              }
              else if (secret <= 250){
                  f38();
              }
              else if (secret <= 260){
                  f39();
              }
              else if (secret <= 265) {
                  f40();
              }
              else if (secret <= 270) {
                  f41();
              }
              else if (secret <= 275){
                  f42();
              }
              else if (secret <= 280){
                  f43();
              }
              else if (secret <= 285) {
                  f44();
              }
              else if (secret <= 290){
                  f45();
              }
              else if (secret <= 295){
                  f46();
              }
              else if (secret <= 300){
                  f47();
              }
              else if (secret <= 310) {
                  f48();
              }
              else if (secret <= 315){
                  f49();
              }
              else if (secret <= 330){
                  f50();
              }
              else if (secret <= 335) {
                  f51();
              }
              else if (secret <= 340){
                  f52();
              }
              else if (secret <= 345){
                  f53();
              }
              else if (secret <= 350){
                  f54();
              }
              else if (secret <= 355) {
                  f55();
              }
              else if (secret <= 360) {
                  f56();
              }
              else if (secret <= 370){
                  f57();
              }
              else if (secret <= 375){
                  f58();
              }
              else if (secret <= 380) {
                  f59();
              }
              else if (secret <= 385){
                  f60();
              }
              else if (secret <= 390){
                  f61();
              }
              else if (secret <= 395){
                  f62();
              }
              else
              {
                  f63();
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
                File file = new File("B_L_5_time.csv");
                if (!file.exists()) {
                    file.createNewFile();
                }
                pw = new FileWriter(file.getAbsoluteFile(), true);
                pw.append(String.valueOf(N)+','+String.valueOf(f0)+','+String.valueOf(f1)+','+String.valueOf(f2)+','+String.valueOf(f3)+','+String.valueOf(f4)+','+String.valueOf(f5)+','+String.valueOf(f6)+','+String.valueOf(f7)+','+String.valueOf(f8)+','+String.valueOf(f9)+','+String.valueOf(f10)+','+String.valueOf(f11)+','+String.valueOf(f12)+','+String.valueOf(f13)+','+String.valueOf(f14)+','+String.valueOf(f15)+','+String.valueOf(f16)+','+String.valueOf(f17)+','+String.valueOf(f18)+','+String.valueOf(f19)+','+String.valueOf(f20)+','+String.valueOf(f21)+','+String.valueOf(f22)+','+String.valueOf(f23)+','+String.valueOf(f24)+','+String.valueOf(f25)+','+String.valueOf(f26)+','+String.valueOf(f27)+','+String.valueOf(f28)+','+String.valueOf(f29)+','+String.valueOf(f30)+','+String.valueOf(f31)+','+String.valueOf(f32)+','+String.valueOf(f33)+','+String.valueOf(f34)+','+String.valueOf(f35)+','+String.valueOf(f36)+','+String.valueOf(f37)+','+String.valueOf(f38)+','+String.valueOf(f39)+','+String.valueOf(f40)+','+String.valueOf(f41)+','+String.valueOf(f42)+','+String.valueOf(f43)+','+String.valueOf(f44)+','+String.valueOf(f45)+','+String.valueOf(f46)+','+String.valueOf(f47)+','+String.valueOf(f48)+','+String.valueOf(f49)+','+String.valueOf(f50)+','+String.valueOf(f51)+','+String.valueOf(f52)+','+String.valueOf(f53)+','+String.valueOf(f54)+','+String.valueOf(f55)+','+String.valueOf(f56)+','+String.valueOf(f57)+','+String.valueOf(f58)+','+String.valueOf(f59)+','+String.valueOf(f60)+','+String.valueOf(f61)+','+String.valueOf(f62)+','+String.valueOf(f63)+','+String.valueOf(dur));
                pw.append('\n');
                pw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    private static void f0() throws InterruptedException {
        f0 = 1;

    }
    private static void f1() throws InterruptedException {
        f1 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(1);
        }
    }
    private static void f2() throws InterruptedException {
        f2 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(2);
        }
    }
    private static void f3() throws InterruptedException {
        f3 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(3);
        }
    }
    private static void f4() throws InterruptedException {
        f4 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(4);
        }
    }
    private static void f5() throws InterruptedException {
        f5 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(5);
        }
    }
    private static void f6() throws InterruptedException {
        f6 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(6);
        }
    }
    private static void f7() throws InterruptedException {
        f7 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(7);
        }
    }
    private static void f8() throws InterruptedException {
        f8 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(8);
        }
    }
    private static void f9() throws InterruptedException {
        f9 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(9);
        }
    }
    private static void f10() throws InterruptedException {
        f10 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(10);
        }
    }
    private static void f11() throws InterruptedException {
        f11 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(11);
        }
    }
    private static void f12() throws InterruptedException {
        f12 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(12);
        }
    }
    private static void f13() throws InterruptedException {
        f13 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(13);
        }
    }
    private static void f14() throws InterruptedException {
        f14 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(14);
        }
    }
    private static void f15() throws InterruptedException {
        f15 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(15);
        }
    }
    private static void f16() throws InterruptedException {
        f16 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(16);
        }
    }
    private static void f17() throws InterruptedException {
        f17 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(17);
        }
    }
    private static void f18() throws InterruptedException {
        f18 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(18);
        }
    }
    private static void f19() throws InterruptedException {
        f19 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(19);
        }
    }
    private static void f20() throws InterruptedException {
        f20 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(20);
        }
    }
    private static void f21() throws InterruptedException {
        f21 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(21);
        }
    }
    private static void f22() throws InterruptedException {
        f22 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(22);
        }
    }
    private static void f23() throws InterruptedException {
        f23 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(23);
        }
    }
    private static void f24() throws InterruptedException {
        f24 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(24);
        }
    }
    private static void f25() throws InterruptedException {
        f25 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(25);
        }
    }
    private static void f26() throws InterruptedException {
        f26 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(26);
        }
    }
    private static void f27() throws InterruptedException {
        f27 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(27);
        }
    }
    private static void f28() throws InterruptedException {
        f28 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(28);
        }
    }
    private static void f29() throws InterruptedException {
        f29 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(29);
        }
    }
    private static void f30() throws InterruptedException {
        f30 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(30);
        }
    }
    private static void f31() throws InterruptedException {
        f31 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(31);
        }
    }
    private static void f32() throws InterruptedException {
        f32 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(32);
        }
    }
    private static void f33() throws InterruptedException {
        f33 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(33);
        }
    }
    private static void f34() throws InterruptedException {
        f34 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(34);
        }
    }
    private static void f35() throws InterruptedException {
        f35 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(35);
        }
    }
    private static void f36() throws InterruptedException {
        f36 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(36);
        }
    }
    private static void f37() throws InterruptedException {
        f37 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(37);
        }
    }
    private static void f38() throws InterruptedException {
        f38 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(38);
        }
    }
    private static void f39() throws InterruptedException {
        f39 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(39);
        }
    }
    private static void f40() throws InterruptedException {
        f40 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(40);
        }
    }
    private static void f41() throws InterruptedException {
        f41 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(41);
        }
    }
    private static void f42() throws InterruptedException {
        f42 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(42);
        }
    }
    private static void f43() throws InterruptedException {
        f43 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(43);
        }
    }
    private static void f44() throws InterruptedException {
        f44 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(44);
        }
    }
    private static void f45() throws InterruptedException {
        f45 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(45);
        }
    }
    private static void f46() throws InterruptedException {
        f46 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(46);
        }
    }
    private static void f47() throws InterruptedException {
        f47 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(47);
        }
    }
    private static void f48() throws InterruptedException {
        f48 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(48);
        }
    }
    private static void f49() throws InterruptedException {
        f49 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(49);
        }
    }
    private static void f50() throws InterruptedException {
        f50 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(50);
        }
    }
    private static void f51() throws InterruptedException {
        f51 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(51);
        }
    }
    private static void f52() throws InterruptedException {
        f52 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(52);
        }
    }
    private static void f53() throws InterruptedException {
        f53 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(53);
        }
    }
    private static void f54() throws InterruptedException {
        f54 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(54);
        }
    }
    private static void f55() throws InterruptedException {
        f55 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(55);
        }
    }
    private static void f56() throws InterruptedException {
        f56 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(56);
        }
    }
    private static void f57() throws InterruptedException {
        f57 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(57);
        }
    }
    private static void f58() throws InterruptedException {
        f58 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(58);
        }
    }
    private static void f59() throws InterruptedException {
        f59 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(59);
        }
    }
    private static void f60() throws InterruptedException {
        f60 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(60);
        }
    }
    private static void f61() throws InterruptedException {
        f61 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(61);
        }
    }
    private static void f62() throws InterruptedException {
        f62 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(62);
        }
    }
    private static void f63() throws InterruptedException {
        f63 = 1;
        for(int i = 0; i < N;i++){
            Thread.sleep(63);
        }
    }
}

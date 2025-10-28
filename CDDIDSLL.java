package mydt;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.core.Utils;
import moa.streams.ArffFileStream;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * 动态阈值:衰减因子递归均值版-窗口精确-可以输出结果到txt文件----2024.3.26
 * 预测确定性新公式--3.0
 * 新加权
 * 新类不平衡比率
 * 新动态系数
 * 可做消融:
 * 消融1:随机请求
 * 消融2:主动请求
 * 消融3:不加权
 * 消融4:固定阈值
 * 消融5:随机请求+不加权+固定阈值
 * 消融6:主动请求+不加权+固定阈值
 */
public class CDDIDSLL {
    //***分类器对象
    private Classifier learner;
    //初始数据流对象
    private ArffFileStream stream;
    //初始训练样本个数
    private int linitLearnSize;
    //数据流类别个数
    private int numClass;
    //类不平衡比率
    private double[] classRate;
    //定义每个类的不确定性阈值数组
    private double[] uncertaintyThreshold;
    //记录每个类下的熵，计算平均熵
    private double[] classShangSum;
    private double[] classShangNum;
    //间隔系数
    private double jge;
    //每个类的最大熵
    private double[] maxShangs;
    //确定性的调节参数
    private double gama;
    //重新学习状态
    private boolean reLearn;
    //动态系数
    private double df;
    //指数加权移动平均调节参数
    private double emwa;

    public CDDIDSLL() {
        resetLearning();
    }

    //初始化
    public void resetLearning() {
        String arffFileNameDive = "D:\\zhuomian\\drift.arff";
        this.learner = new NaiveBayes();
//        this.learner = new HoeffdingTree();
        this.stream = new ArffFileStream(arffFileNameDive, -1);
        //数据流准备使用
        stream.prepareForUse();
        //把数据流信息给分类器对象
        learner.setModelContext(stream.getHeader());
        //分类器对象准备使用
        learner.prepareForUse();
        //类别个数
        this.numClass = stream.getHeader().numClasses();
        //定义一个记录每个类比率的数组
        this.classRate = new double[numClass];
        //定义每个类的不确定性阈值数组
        this.uncertaintyThreshold = new double[numClass];
        //记录每个类下的熵，计算平均熵
        this.classShangSum = new double[numClass];
        this.classShangNum = new double[numClass];
        this.maxShangs = new double[numClass];
        this.linitLearnSize = 1000;
        this.reLearn = true;
        this.df = 1;
        this.jge = 0.40;
        this.emwa = 0.1;
        this.gama = 0.8;
    }

    public void run(int numInstances) throws IOException {
        Runtime runtime = Runtime.getRuntime();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        Instant start = Instant.now();
        int LWsize = 1000;
        Queue<Integer> labelWindow = new ArrayDeque<>();
        int subtimes = 0;
        int updatetimes = 0;
        int[] classNumMo = new int[numClass];
        int SWsize = 500;
        LinkedList<int[]> shortWindow = new LinkedList<>();
        int[] tempQueueArray;
        int getAccTime = 0;
        double delta = 0.005;
        double yuzhib = Math.sqrt(1.0 / (2 * SWsize) * Math.log(1.0 / delta));
        double windowAccMax = 0;
        int errTemp = 0;
        int setErrTimes = 100;
        double ruselt = 0;
        Queue<Integer> errWindow = new LinkedList<>();
        double maxErrDis = 0;
        double[] defInit;
        double sumInit;
        int classMaxIdex;
        double classMaxValue;
        double unValue;
        int numberSamplesCorrect = 0;
        int acctimes = 0;
        int accNums = 0;
        double accSums = 0;
        int accWsize = 1000;
        Queue<Integer> accWindow = new ArrayDeque<>();
        List<Double> arrayList = new ArrayList<>();
        int numberSamples = 0;
        int checkNum = 0;
        int tempCheckNum = 0;
        int driftTimes = 0;
        int numxun = 0;
        int[] classNum = new int[numClass];

        Random random = new Random();

        //存储y
        List<Double> yList = new ArrayList<>();

        while (stream.hasMoreInstances() && numberSamples < numInstances) {
            Instance trainInst = stream.nextInstance().getData();
            numberSamples++;
            acctimes++;
            if (learner.correctlyClassifies(trainInst)) {
                numberSamplesCorrect++;
            }
            if (accWindow.size() == accWsize) {
                accWindow.poll();
            }
            if (learner.correctlyClassifies(trainInst)) {
                numberSamplesCorrect++;
                accWindow.offer(1);
            } else {
                accWindow.offer(0);
            }
            if (acctimes == 1000) {
                acctimes = 0;
                accNums++;
                int sum = calculateSum(accWindow);
                double acc = (double) sum / accWindow.size();
                accSums += acc;
                arrayList.add(acc);
                yList.add(acc);
            }
            //重学习
            if (reLearn) {
                numxun++;
                learner.trainOnInstance(trainInst);
                classNum[(int) trainInst.classValue()] += 1;

                defInit = learner.getVotesForInstance(trainInst);
                sumInit = 0;
                for (double prob : defInit) {
                    sumInit += prob;
                }
                if (sumInit > 0.0) {
                    for (int i = 0; i < defInit.length; i++) {
                        defInit[i] /= sumInit;
                    }
                }
                classMaxIdex = Utils.maxIndex(defInit);
                classMaxValue = defInit[classMaxIdex];
                unValue = 1 - queDing(defInit, classMaxValue);
                if (classShangNum[classMaxIdex] == 0) {
                    classShangSum[classMaxIdex] = unValue;
                } else {
                    classShangSum[classMaxIdex] = (1 - emwa) * classShangSum[classMaxIdex] + emwa * unValue;
                }
                classShangNum[classMaxIdex] += 1;
                if (learner.correctlyClassifies(trainInst)) {
                    if (unValue > maxShangs[classMaxIdex]) {
                        maxShangs[classMaxIdex] = unValue;
                    }
                }
                uncertaintyThreshold[classMaxIdex] = maxShangs[classMaxIdex] - jge * (maxShangs[classMaxIdex] -
                        classShangSum[classMaxIdex]);
                if (numxun == linitLearnSize) {
                    numxun = 0;
                    for (int i = 0; i < classRate.length; i++) {
                        classRate[i] = classNum[i] / 1000.0;
                    }
                    System.out.println(Arrays.toString(classRate));
                    System.out.println(Arrays.toString(uncertaintyThreshold));
                    reLearn = false;
                }
            } else {
                defInit = learner.getVotesForInstance(trainInst);
                sumInit = 0;
                for (double prob : defInit) {
                    sumInit += prob;
                }
                if (sumInit > 0.0) {
                    for (int i = 0; i < defInit.length; i++) {
                        defInit[i] /= sumInit;
                    }
                }
                classMaxIdex = Utils.maxIndex(defInit);
                classMaxValue = defInit[classMaxIdex];
                unValue = 1 - queDing(defInit, classMaxValue);
                if (classShangNum[classMaxIdex] == 0) {
                    classShangSum[classMaxIdex] = unValue;
                } else {
                    classShangSum[classMaxIdex] = (1 - emwa) * classShangSum[classMaxIdex] + emwa * unValue;
                }
                classShangNum[classMaxIdex] += 1;
                if (getAccTime >= 1) {
                    getAccTime = 0;
                    if (tempCheckNum >= 500) {
                        double windowErrorSum = 0;
                        double windoAllSum = 0;
                        assert shortWindow.peek() != null;
                        int timemin = shortWindow.peek()[2];
                        assert shortWindow.peekLast() != null;
                        int timemax = shortWindow.peekLast()[2];
                        int timecha = timemax - timemin;
                        for (int[] elemet : shortWindow) {
                            double temps0 = ((double) (elemet[2]-timemin)/timecha)+(1-classRate[elemet[0]]);
                            windowErrorSum += elemet[1] * temps0;
                            windoAllSum += temps0;
                        }
                        double windowAcc = 1 - (windowErrorSum / windoAllSum);
                        if (windowAcc > windowAccMax) {
                            windowAccMax = windowAcc;
                        }
                        double chaACC = windowAccMax - windowAcc;
                        if (!errWindow.isEmpty()) {
                            int sum = 0;
                            int count = 0;
                            for (int num : errWindow) {
                                sum += num;
                                count++;
                            }
                            ruselt = (double) sum / count;
                            if (ruselt > maxErrDis) {
                                maxErrDis = ruselt;
                            }
                        }
                        yuzhib = Math.sqrt((df * (ruselt/maxErrDis)) / (2 * SWsize) * Math.log(1.0 / delta));
                        if (chaACC > yuzhib) {
                            driftTimes++;
                            System.out.println("检测到漂移，检测点为" + numberSamples);
                            tempCheckNum = 0;
                            windowAccMax = 0;
                            maxErrDis = 0;
                            subtimes = 0;
                            updatetimes = 0;
                            this.classRate = new double[numClass];
                            classNum = new int[numClass];
                            this.uncertaintyThreshold = new double[numClass];
                            this.classShangSum = new double[numClass];
                            this.classShangNum = new double[numClass];
                            this.maxShangs = new double[numClass];
                            classNumMo = new int[numClass];
                            labelWindow.clear();
                            this.learner.resetLearning();
                            this.reLearn = true;
                            continue;
                        }
                    }
                }
                if (((double) checkNum / numberSamples) <= 0.25) {
                    double randomProbability = random.nextDouble();
                    if (unValue >= uncertaintyThreshold[classMaxIdex]) {
                        checkNum++;
                        tempCheckNum++;
                        tempQueueArray = new int[3];
                        if (shortWindow.size() == SWsize) {
                            shortWindow.poll();
                        }
                        if (learner.correctlyClassifies(trainInst)) {
                            if (unValue > maxShangs[classMaxIdex]) {
                                maxShangs[classMaxIdex] = unValue;
                            }
                            tempQueueArray[0] = (int) trainInst.classValue();
                            tempQueueArray[2] = numberSamples;
                            shortWindow.offer(tempQueueArray);
                        } else {
                            tempQueueArray[0] = (int) trainInst.classValue();
                            tempQueueArray[1] = 1;
                            tempQueueArray[2] = numberSamples;
                            shortWindow.offer(tempQueueArray);
                            if (errWindow.size() == setErrTimes) {
                                errWindow.poll();
                            }
                            errWindow.offer(checkNum - errTemp);
                            errTemp = checkNum;
                        }
                        getAccTime++;
                        if (labelWindow.size() == LWsize) {
                            int i = labelWindow.poll();
                            if (i != -1)
                                classNumMo[i] -= 1;
                        }
                        labelWindow.offer(-1);
                        learner.trainOnInstance(trainInst);
                    }
                    else if (randomProbability<=0.2) {
                        learner.trainOnInstance(trainInst);
                        checkNum++;
                        tempCheckNum++;
                        subtimes = 0;
                        getAccTime++;
                        if (labelWindow.size() == LWsize) {
                            int i = labelWindow.poll();
                            if (i != -1)
                                classNumMo[i] -= 1;
                        }
                        labelWindow.offer((int) trainInst.classValue());
                        classNumMo[(int) trainInst.classValue()] += 1;
                        tempQueueArray = new int[3];
                        if (shortWindow.size() == SWsize) {
                            shortWindow.poll();
                        }
                        if (learner.correctlyClassifies(trainInst)) {
                            if (unValue > maxShangs[classMaxIdex]) {
                                maxShangs[classMaxIdex] = unValue;
                            }
                            tempQueueArray[0] = (int) trainInst.classValue();
                            tempQueueArray[2] = numberSamples;
                            shortWindow.offer(tempQueueArray);
                        } else {
                            tempQueueArray[0] = (int) trainInst.classValue();
                            tempQueueArray[1] = 1;
                            tempQueueArray[2] = numberSamples;
                            shortWindow.offer(tempQueueArray);
                            if (errWindow.size() == setErrTimes) {
                                errWindow.poll();
                            }
                            errWindow.offer(checkNum - errTemp);
                            errTemp = checkNum;
                        }

                    } else {
                        if (labelWindow.size() == LWsize) {
                            int i = labelWindow.poll();
                            if (i != -1)
                                classNumMo[i] -= 1;
                        }
                        labelWindow.offer(-1);

                        subtimes++;
                    }
                } else {
                    if (labelWindow.size() == LWsize) {
                        int i = labelWindow.poll();
                        if (i != -1)
                            classNumMo[i] -= 1;
                    }
                    labelWindow.offer(-1);
                }
                uncertaintyThreshold[classMaxIdex] = maxShangs[classMaxIdex] - jge * (maxShangs[classMaxIdex] -
                        classShangSum[classMaxIdex]);
                updatetimes++;
                if (updatetimes == 1000) {
                    updatetimes = 0;
                    int count = countOccurrences(labelWindow, -1);
                    for (int i = 0; i < classRate.length; i++) {
                        classRate[i] = (double) classNumMo[i] / (labelWindow.size() - count);
                    }
                }
            }
        }
        double average = calculateAverage(arrayList);
        System.out.println("平均值：" + average);
        double variance = calculateVariance(arrayList, average);
        System.out.println("方差：" + variance);
        Instant end = Instant.now();
        Duration duration = Duration.between(start, end);
        System.out.println("执行时间：" + duration.toMillis() + " 毫秒");
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = memoryAfter - memoryBefore;
        System.out.println("内存使用：" + memoryUsed + " 字节");
        BufferedWriter yOut = null;
    }
    private double queDing(double[] gaiLu, double maxgaiLu) {
        Arrays.sort(gaiLu);
        reverseArray(gaiLu);
        if (gaiLu.length == 2) {
            double tempQue = (gama * maxgaiLu) + (1 - gama) * (gaiLu[0] - gaiLu[1]);
            return (tempQue - gama * 0.5) / (1 - gama * 0.5);
        }
        double maxQue = gama * 1 + (1 - gama) * 0.5;
        double minQue = gama * (1.0 / gaiLu.length);
        int i;
        double sum = 0;
        for (i = 0; i < gaiLu.length - 1; i++) {
            if (i == gaiLu.length - 2)
                sum += ((gaiLu[i] - gaiLu[i + 1]) * (1.0 / Math.pow(2, i)));
            else
                sum += ((gaiLu[i] - gaiLu[i + 1]) * (1.0 / Math.pow(2, i + 1)));
        }
        return (((gama * maxgaiLu) + (1 - gama) * sum) - minQue) / (maxQue - minQue);
    }
    public static int countOccurrences(Queue<Integer> queue, int target) {
        int count = 0;
        for (Integer num : queue) {
            if (num == target) {
                count++;
            }
        }
        return count;
    }
    private static int calculateSum(Queue<Integer> queue) {
        int sum = 0;
        for (Integer num : queue) {
            sum += num;
        }
        return sum;
    }
    private static double calculateAverage(List<Double> list) {
        return list.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    private static double calculateVariance(List<Double> list, double mean) {
        return list.stream().mapToDouble(num -> Math.pow(num - mean, 2)).average().orElse(0.0);
    }
    public void reverseArray(double[] array) {
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            double temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            left++;
            right--;
        }
    }

}


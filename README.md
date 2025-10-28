README for CDDIDSLL

📌 项目简介:  
CDDIDSLL是一个基于 MOA (Massive Online Analysis) 框架的概念漂移检测方法实验代码。  

📂 依赖环境:  
主要依赖  
Java 8+  
MOA (Massive Online Analysis):https://moa.cms.waikato.ac.nz/  
数据格式：.arff文件  
使用的分类器  
默认：NaiveBayes  
（可以切换为 HoeffdingTree 等 MOA 提供的流式分类器  this.learner = new HoeffdingTree();）  

⚙️ 文件说明:  
CDDIDSLL.java：主类，包含流式学习与漂移检测逻辑。  
drift.arff：输入数据流文件（需用户提供，示例路径：D:\zhuomian\drift.arff）。  

🚀 运行方法:  
在 main 方法中调用：  
public static void main(String[] args) throws Exception {  
    CDDIDSLL exp = new CDDIDSLL();  
    exp.run(100000); // 设置运行样本数  
}  

package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import org.kohsuke.args4j.CmdLineException;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.*;
import java.io.*;

import JavaExtractor.Subtokenizer;




public class App {
    private static CommandLineValues s_CommandLineValues;

    public static void main(String[] args) throws IOException {
        try {
            s_CommandLineValues = new CommandLineValues(args);
        } catch (CmdLineException e) {
            e.printStackTrace();
            return;
        }

        ExtractFeaturesTask extractFeaturesTask = new ExtractFeaturesTask(s_CommandLineValues);

        String valid_save_path="../Raw/ast/";
        File file=new File(valid_save_path);
        if(!file.exists()) file.mkdirs();

        String[] modes={"train","test","valid"};
        for(String mode:modes) {
            String data_path = "../Raw/"+mode+"_src.json";
            Map data = readData(data_path);
            JSONArray jsonArray = new JSONArray();
            int count = 0;
            for (Object obj : data.keySet()) {
                if (count % 1000 == 0)
                    System.out.println("Processing count: " + count);
                String fid = obj.toString(), src = data.get(obj).toString();
                jsonArray = extractFeaturesTask.process(src, fid, jsonArray);
                count++;
            }
            String jsonOutput = jsonArray.toJSONString();
            String save_path = "../Raw/ast/"+mode+"_ast.json";
            try {
                File fileText = new File(save_path);
                FileWriter fileWriter = new FileWriter(fileText);
                fileWriter.write(jsonOutput);
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("IOException");
            }
        }

    }

    public static Map readData(String path) {
        String data="";
        try { data = readJson(path); } catch (IOException e) { e.printStackTrace(); }
        Map codes = JSON.parseObject(data);
        return codes;
    }
    public static String readJson(String path) throws IOException {
        StringBuffer strbuffer = new StringBuffer();
        File myFile = new File(path);
        if (!myFile.exists()) { System.err.println("Can't Find " + path); }
        try {
            FileInputStream fis = new FileInputStream(path);
            InputStreamReader inputStreamReader = new InputStreamReader(fis, "UTF-8");
            BufferedReader in  = new BufferedReader(inputStreamReader);
            String str;
            while ((str = in.readLine()) != null) { strbuffer.append(str); }
            in.close();
        } catch (IOException e) { e.getStackTrace(); }
        return strbuffer.toString();
    }
}

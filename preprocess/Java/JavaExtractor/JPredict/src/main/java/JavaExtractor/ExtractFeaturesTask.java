package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.lang3.StringUtils;

class ExtractFeaturesTask {
    private final CommandLineValues m_CommandLineValues;

    public ExtractFeaturesTask(CommandLineValues commandLineValues) {
        m_CommandLineValues = commandLineValues;
    }

    public JSONArray process(String code, String fid, JSONArray jsonArray) {
        FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues);
        jsonArray=featureExtractor.extractAST(code,fid,jsonArray);
        return jsonArray;
    }
}

package model;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.Scanner;


/*
Inspired by code from https://crunchify.com/how-to-read-json-object-from-file-in-java/
 */

public class JSONReader {

    private JSONObject jsonObject;

    public JSONReader(String filepath) {
        try {
//            String text = new String(Files.readAllBytes(Paths.get(filepath)));
//            String text = new String(Files.readAllBytes(filepath), StandardCharsets.UTF_8);
//            String text = StandardCharsets.UTF_8.decode(ByteBuffer.wrap(Files.readAllBytes(filepath)));
            String text = new Scanner(new File(filepath)).useDelimiter("\\A").next();
            jsonObject = new JSONObject(text);
        } catch (Exception e) {
            System.out.println("failed to read JSON");
            // tell user if fails (likely because wrong path or wrong file type)
            e.printStackTrace();
        }
    }

    public JSONObject getData() {
        return jsonObject;
    }

    // example usage for looping over parameters in a file from .templates (top level list)
    public static void main(String[] args) {
        // NOTE: "/" can be used on any OS in Java! HYPE!
        JSONObject data = new JSONReader(".templates/CorTec.json").getData();
        for (Object item: (JSONArray) data.get("data")) {
            JSONObject itemObject = (JSONObject) item;
            System.out.println("expression: " + itemObject.get("expression"));
            System.out.println("name: " + itemObject.get("name"));
            System.out.println("");
        }
    }
}
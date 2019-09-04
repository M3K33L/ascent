package model;

import java.util.HashMap;
import java.util.Map;

class IdentifierManager {

    private Map<String, Integer> identifiers;

    public IdentifierManager() {
        identifiers = new HashMap<String, Integer>();
    }

    public String next(String key) {
        // default next index to 1 (assume first call of key)
        int nextIndex = 1;

        // if the key already exists, set
        if (identifiers.containsKey(key)) {
            nextIndex = identifiers.get(key) + 1;
        }

        // update identifiers index
        identifiers.put(key, nextIndex);

        // return String version (i.e. "keyN")
        return key + Integer.toString(nextIndex);
    }
}
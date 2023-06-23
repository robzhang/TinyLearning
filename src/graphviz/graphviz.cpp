//
// Created by Fangbo Zhang on 2023/6/8.
//
#include <string>
#include <memory>
#include <set>
#include <map>
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "core/variable.h"
#include "core/function.h"

using namespace std;

namespace TinyLearning {
    template<class... T>
    string format(const char *fmt, const T&...t)
    {
        const auto len = snprintf(nullptr, 0, fmt, t...);

        string r;
        r.resize(static_cast<size_t>(len) + 1);
        snprintf(&r.front(), len + 1, fmt, t...);
        r.resize(static_cast<size_t>(len));

        return r;
    }

    template <typename F>
    static string drawVariable(const shared_ptr<Variable>& variable, F getId) {
        const char* dot_var_ = "%d [label=\"%s\", color=orange, style=filled]\n";

        int id = getId((uintptr_t) variable.get());

        return format(dot_var_, id, variable->Name().c_str());
    }

    template <typename F>
    static string drawFunction(const shared_ptr<Function>& f, F getId) {
        const char* dot_func_ = "%d [label=\"%s\", color=lightblue, style=filled, shape=box]\n";

        int fid = getId((uintptr_t) f.get());

        string txt = format(dot_func_, fid, f->Name());

        const char* edge = "%d -> %d\n";

        for (const auto& x : f->Input()) {
            int xid = getId((uintptr_t) x.get());

            txt += format(edge, xid, fid);
        }

        for (const auto& y : f->Output()) {
            int yid = getId((uintptr_t) y.lock().get());

            txt += format(edge, fid, yid);
        }

        return txt;
    }

    string DrawGraphviz(const shared_ptr<Variable>& v) {
        int id = 0;
        std::map<uintptr_t, int> addr2id;
        auto GetId = [&](uintptr_t addr) mutable ->int {
            auto iter = addr2id.find(addr);
            if (iter != addr2id.end()) {
                return iter->second;
            } else {
                id++;
                addr2id.insert(pair<uintptr_t, int>(addr, id));
                return id;
            }
        };

        std::set<shared_ptr<Function>, Function::Comparator> funcs = {v->Creator()};

        string txt = drawVariable(v, GetId);

        do {
            const auto f = *funcs.begin();
            funcs.erase(f);

            txt += drawFunction(f, GetId);

            for (const auto& x : f->Input()) {
                txt += drawVariable(x, GetId);

                if (x->Creator()) {
                    funcs.insert(x->Creator());
                }
            }
        } while (!funcs.empty());

        return "digraph g {\n" + txt + "}";
    }

    static char* getCWD() {
        return getcwd(nullptr, 0);
    }

    static void writeTextFile(string& filename, string& data) {
        ofstream dotFile(filename);
        dotFile << data;
        dotFile.close();
    }

    void DrawComputationGraph(const shared_ptr<Variable>& v, const string& filename) {
        auto graphviz = DrawGraphviz(v);

        auto cwd = getCWD();
        if (cwd == nullptr) {
            cout << "getCWD error" << endl;
            return;
        }

        auto graphvizFilename = string(cwd) + "/" + v->Name() + ".dot";
        writeTextFile(graphvizFilename, graphviz);

        const char* cmdFormat = "dot %s -T %s -o %s";
        auto cmd = format(cmdFormat, graphvizFilename.c_str(), "png", filename.c_str());

        auto ret = std::system(cmd.c_str());
        if (ret == -1) {
            cout << "system() error!" << endl;
        }

        free(cwd);
    }
}

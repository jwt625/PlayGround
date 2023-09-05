(() => {
    var e = {
            76: function (e, t, n) {
                ! function (e) {
                    "use strict";

                    function t(e, t) {
                        e.super_ = t, e.prototype = Object.create(t.prototype, {
                            constructor: {
                                value: e,
                                enumerable: !1,
                                writable: !0,
                                configurable: !0
                            }
                        })
                    }

                    function r(e, t) {
                        Object.defineProperty(this, "kind", {
                            value: e,
                            enumerable: !0
                        }), t && t.length && Object.defineProperty(this, "path", {
                            value: t,
                            enumerable: !0
                        })
                    }

                    function i(e, t, n) {
                        i.super_.call(this, "E", e), Object.defineProperty(this, "lhs", {
                            value: t,
                            enumerable: !0
                        }), Object.defineProperty(this, "rhs", {
                            value: n,
                            enumerable: !0
                        })
                    }

                    function o(e, t) {
                        o.super_.call(this, "N", e), Object.defineProperty(this, "rhs", {
                            value: t,
                            enumerable: !0
                        })
                    }

                    function a(e, t) {
                        a.super_.call(this, "D", e), Object.defineProperty(this, "lhs", {
                            value: t,
                            enumerable: !0
                        })
                    }

                    function s(e, t, n) {
                        s.super_.call(this, "A", e), Object.defineProperty(this, "index", {
                            value: t,
                            enumerable: !0
                        }), Object.defineProperty(this, "item", {
                            value: n,
                            enumerable: !0
                        })
                    }

                    function u(e, t, n) {
                        var r = e.slice((n || t) + 1 || e.length);
                        return e.length = t < 0 ? e.length + t : t, e.push.apply(e, r), e
                    }

                    function c(e) {
                        var t = void 0 === e ? "undefined" : j(e);
                        return "object" !== t ? t : e === Math ? "math" : null === e ? "null" : Array.isArray(e) ? "array" : "[object Date]" === Object.prototype.toString.call(e) ? "date" : "function" == typeof e.toString && /^\/.*\//.test(e.toString()) ? "regexp" : "object"
                    }

                    function l(e, t, n, r, d, f, p) {
                        p = p || [];
                        var h = (d = d || []).slice(0);
                        if (void 0 !== f) {
                            if (r) {
                                if ("function" == typeof r && r(h, f)) return;
                                if ("object" === (void 0 === r ? "undefined" : j(r))) {
                                    if (r.prefilter && r.prefilter(h, f)) return;
                                    if (r.normalize) {
                                        var g = r.normalize(h, f, e, t);
                                        g && (e = g[0], t = g[1])
                                    }
                                }
                            }
                            h.push(f)
                        }
                        "regexp" === c(e) && "regexp" === c(t) && (e = e.toString(), t = t.toString());
                        var m = void 0 === e ? "undefined" : j(e),
                            v = void 0 === t ? "undefined" : j(t),
                            y = "undefined" !== m || p && p[p.length - 1].lhs && p[p.length - 1].lhs.hasOwnProperty(f),
                            b = "undefined" !== v || p && p[p.length - 1].rhs && p[p.length - 1].rhs.hasOwnProperty(f);
                        if (!y && b) n(new o(h, t));
                        else if (!b && y) n(new a(h, e));
                        else if (c(e) !== c(t)) n(new i(h, e, t));
                        else if ("date" === c(e) && e - t != 0) n(new i(h, e, t));
                        else if ("object" === m && null !== e && null !== t)
                            if (p.filter((function (t) {
                                    return t.lhs === e
                                })).length) e !== t && n(new i(h, e, t));
                            else {
                                if (p.push({
                                        lhs: e,
                                        rhs: t
                                    }), Array.isArray(e)) {
                                    var w;
                                    for (e.length, w = 0; w < e.length; w++) w >= t.length ? n(new s(h, w, new a(void 0, e[w]))) : l(e[w], t[w], n, r, h, w, p);
                                    for (; w < t.length;) n(new s(h, w, new o(void 0, t[w++])))
                                } else {
                                    var k = Object.keys(e),
                                        x = Object.keys(t);
                                    k.forEach((function (i, o) {
                                        var a = x.indexOf(i);
                                        a >= 0 ? (l(e[i], t[i], n, r, h, i, p), x = u(x, a)) : l(e[i], void 0, n, r, h, i, p)
                                    })), x.forEach((function (e) {
                                        l(void 0, t[e], n, r, h, e, p)
                                    }))
                                }
                                p.length = p.length - 1
                            }
                        else e !== t && ("number" === m && isNaN(e) && isNaN(t) || n(new i(h, e, t)))
                    }

                    function d(e, t, n, r) {
                        return r = r || [], l(e, t, (function (e) {
                            e && r.push(e)
                        }), n), r.length ? r : void 0
                    }

                    function f(e, t, n) {
                        if (n.path && n.path.length) {
                            var r, i = e[t],
                                o = n.path.length - 1;
                            for (r = 0; r < o; r++) i = i[n.path[r]];
                            switch (n.kind) {
                                case "A":
                                    f(i[n.path[r]], n.index, n.item);
                                    break;
                                case "D":
                                    delete i[n.path[r]];
                                    break;
                                case "E":
                                case "N":
                                    i[n.path[r]] = n.rhs
                            }
                        } else switch (n.kind) {
                            case "A":
                                f(e[t], n.index, n.item);
                                break;
                            case "D":
                                e = u(e, t);
                                break;
                            case "E":
                            case "N":
                                e[t] = n.rhs
                        }
                        return e
                    }

                    function p(e, t, n) {
                        if (e && t && n && n.kind) {
                            for (var r = e, i = -1, o = n.path ? n.path.length - 1 : 0; ++i < o;) void 0 === r[n.path[i]] && (r[n.path[i]] = "number" == typeof n.path[i] ? [] : {}), r = r[n.path[i]];
                            switch (n.kind) {
                                case "A":
                                    f(n.path ? r[n.path[i]] : r, n.index, n.item);
                                    break;
                                case "D":
                                    delete r[n.path[i]];
                                    break;
                                case "E":
                                case "N":
                                    r[n.path[i]] = n.rhs
                            }
                        }
                    }

                    function h(e, t, n) {
                        if (n.path && n.path.length) {
                            var r, i = e[t],
                                o = n.path.length - 1;
                            for (r = 0; r < o; r++) i = i[n.path[r]];
                            switch (n.kind) {
                                case "A":
                                    h(i[n.path[r]], n.index, n.item);
                                    break;
                                case "D":
                                case "E":
                                    i[n.path[r]] = n.lhs;
                                    break;
                                case "N":
                                    delete i[n.path[r]]
                            }
                        } else switch (n.kind) {
                            case "A":
                                h(e[t], n.index, n.item);
                                break;
                            case "D":
                            case "E":
                                e[t] = n.lhs;
                                break;
                            case "N":
                                e = u(e, t)
                        }
                        return e
                    }

                    function g(e, t, n) {
                        if (e && t && n && n.kind) {
                            var r, i, o = e;
                            for (i = n.path.length - 1, r = 0; r < i; r++) void 0 === o[n.path[r]] && (o[n.path[r]] = {}), o = o[n.path[r]];
                            switch (n.kind) {
                                case "A":
                                    h(o[n.path[r]], n.index, n.item);
                                    break;
                                case "D":
                                case "E":
                                    o[n.path[r]] = n.lhs;
                                    break;
                                case "N":
                                    delete o[n.path[r]]
                            }
                        }
                    }

                    function m(e, t, n) {
                        e && t && l(e, t, (function (r) {
                            n && !n(e, t, r) || p(e, t, r)
                        }))
                    }

                    function v(e) {
                        return "color: " + T[e].color + "; font-weight: bold"
                    }

                    function y(e) {
                        var t = e.kind,
                            n = e.path,
                            r = e.lhs,
                            i = e.rhs,
                            o = e.index,
                            a = e.item;
                        switch (t) {
                            case "E":
                                return [n.join("."), r, "→", i];
                            case "N":
                                return [n.join("."), i];
                            case "D":
                                return [n.join(".")];
                            case "A":
                                return [n.join(".") + "[" + o + "]", a];
                            default:
                                return []
                        }
                    }

                    function b(e, t, n, r) {
                        var i = d(e, t);
                        try {
                            r ? n.groupCollapsed("diff") : n.group("diff")
                        } catch (e) {
                            n.log("diff")
                        }
                        i ? i.forEach((function (e) {
                            var t = e.kind,
                                r = y(e);
                            n.log.apply(n, ["%c " + T[t].text, v(t)].concat(E(r)))
                        })) : n.log("—— no diff ——");
                        try {
                            n.groupEnd()
                        } catch (e) {
                            n.log("—— diff end —— ")
                        }
                    }

                    function w(e, t, n, r) {
                        switch (void 0 === e ? "undefined" : j(e)) {
                            case "object":
                                return "function" == typeof e[r] ? e[r].apply(e, E(n)) : e[r];
                            case "function":
                                return e(t);
                            default:
                                return e
                        }
                    }

                    function k(e) {
                        var t = e.timestamp,
                            n = e.duration;
                        return function (e, r, i) {
                            var o = ["action"];
                            return o.push("%c" + String(e.type)), t && o.push("%c@ " + r), n && o.push("%c(in " + i.toFixed(2) + " ms)"), o.join(" ")
                        }
                    }

                    function x(e, t) {
                        var n = t.logger,
                            r = t.actionTransformer,
                            i = t.titleFormatter,
                            o = void 0 === i ? k(t) : i,
                            a = t.collapsed,
                            s = t.colors,
                            u = t.level,
                            c = t.diff,
                            l = void 0 === t.titleFormatter;
                        e.forEach((function (i, d) {
                            var f = i.started,
                                p = i.startedTime,
                                h = i.action,
                                g = i.prevState,
                                m = i.error,
                                v = i.took,
                                y = i.nextState,
                                k = e[d + 1];
                            k && (y = k.prevState, v = k.started - f);
                            var x = r(h),
                                A = "function" == typeof a ? a((function () {
                                    return y
                                }), h, i) : a,
                                _ = C(p),
                                O = s.title ? "color: " + s.title(x) + ";" : "",
                                I = ["color: gray; font-weight: lighter;"];
                            I.push(O), t.timestamp && I.push("color: gray; font-weight: lighter;"), t.duration && I.push("color: gray; font-weight: lighter;");
                            var P = o(x, _, v);
                            try {
                                A ? s.title && l ? n.groupCollapsed.apply(n, ["%c " + P].concat(I)) : n.groupCollapsed(P) : s.title && l ? n.group.apply(n, ["%c " + P].concat(I)) : n.group(P)
                            } catch (e) {
                                n.log(P)
                            }
                            var S = w(u, x, [g], "prevState"),
                                j = w(u, x, [x], "action"),
                                E = w(u, x, [m, g], "error"),
                                D = w(u, x, [y], "nextState");
                            if (S)
                                if (s.prevState) {
                                    var T = "color: " + s.prevState(g) + "; font-weight: bold";
                                    n[S]("%c prev state", T, g)
                                } else n[S]("prev state", g);
                            if (j)
                                if (s.action) {
                                    var R = "color: " + s.action(x) + "; font-weight: bold";
                                    n[j]("%c action    ", R, x)
                                } else n[j]("action    ", x);
                            if (m && E)
                                if (s.error) {
                                    var q = "color: " + s.error(m, g) + "; font-weight: bold;";
                                    n[E]("%c error     ", q, m)
                                } else n[E]("error     ", m);
                            if (D)
                                if (s.nextState) {
                                    var M = "color: " + s.nextState(y) + "; font-weight: bold";
                                    n[D]("%c next state", M, y)
                                } else n[D]("next state", y);
                            c && b(g, y, n, A);
                            try {
                                n.groupEnd()
                            } catch (e) {
                                n.log("—— log end ——")
                            }
                        }))
                    }

                    function A() {
                        var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                            t = Object.assign({}, R, e),
                            n = t.logger,
                            r = t.stateTransformer,
                            i = t.errorTransformer,
                            o = t.predicate,
                            a = t.logErrors,
                            s = t.diffPredicate;
                        if (void 0 === n) return function () {
                            return function (e) {
                                return function (t) {
                                    return e(t)
                                }
                            }
                        };
                        if (e.getState && e.dispatch) return console.error("[redux-logger] redux-logger not installed. Make sure to pass logger instance as middleware:\n// Logger with default options\nimport { logger } from 'redux-logger'\nconst store = createStore(\n  reducer,\n  applyMiddleware(logger)\n)\n// Or you can create your own logger with custom options http://bit.ly/redux-logger-options\nimport createLogger from 'redux-logger'\nconst logger = createLogger({\n  // ...options\n});\nconst store = createStore(\n  reducer,\n  applyMiddleware(logger)\n)\n"),
                            function () {
                                return function (e) {
                                    return function (t) {
                                        return e(t)
                                    }
                                }
                            };
                        var u = [];
                        return function (e) {
                            var n = e.getState;
                            return function (e) {
                                return function (c) {
                                    if ("function" == typeof o && !o(n, c)) return e(c);
                                    var l = {};
                                    u.push(l), l.started = S.now(), l.startedTime = new Date, l.prevState = r(n()), l.action = c;
                                    var d = void 0;
                                    if (a) try {
                                        d = e(c)
                                    } catch (e) {
                                        l.error = i(e)
                                    } else d = e(c);
                                    l.took = S.now() - l.started, l.nextState = r(n());
                                    var f = t.diff && "function" == typeof s ? s(n, c) : t.diff;
                                    if (x(u, Object.assign({}, t, {
                                            diff: f
                                        })), u.length = 0, l.error) throw l.error;
                                    return d
                                }
                            }
                        }
                    }
                    var _, O, I = function (e, t) {
                            return new Array(t + 1).join(e)
                        },
                        P = function (e, t) {
                            return I("0", t - e.toString().length) + e
                        },
                        C = function (e) {
                            return P(e.getHours(), 2) + ":" + P(e.getMinutes(), 2) + ":" + P(e.getSeconds(), 2) + "." + P(e.getMilliseconds(), 3)
                        },
                        S = "undefined" != typeof performance && null !== performance && "function" == typeof performance.now ? performance : Date,
                        j = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
                            return typeof e
                        } : function (e) {
                            return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                        },
                        E = function (e) {
                            if (Array.isArray(e)) {
                                for (var t = 0, n = Array(e.length); t < e.length; t++) n[t] = e[t];
                                return n
                            }
                            return Array.from(e)
                        },
                        D = [];
                    _ = "object" === (void 0 === n.g ? "undefined" : j(n.g)) && n.g ? n.g : "undefined" != typeof window ? window : {}, (O = _.DeepDiff) && D.push((function () {
                        void 0 !== O && _.DeepDiff === d && (_.DeepDiff = O, O = void 0)
                    })), t(i, r), t(o, r), t(a, r), t(s, r), Object.defineProperties(d, {
                        diff: {
                            value: d,
                            enumerable: !0
                        },
                        observableDiff: {
                            value: l,
                            enumerable: !0
                        },
                        applyDiff: {
                            value: m,
                            enumerable: !0
                        },
                        applyChange: {
                            value: p,
                            enumerable: !0
                        },
                        revertChange: {
                            value: g,
                            enumerable: !0
                        },
                        isConflict: {
                            value: function () {
                                return void 0 !== O
                            },
                            enumerable: !0
                        },
                        noConflict: {
                            value: function () {
                                return D && (D.forEach((function (e) {
                                    e()
                                })), D = null), d
                            },
                            enumerable: !0
                        }
                    });
                    var T = {
                            E: {
                                color: "#2196F3",
                                text: "CHANGED:"
                            },
                            N: {
                                color: "#4CAF50",
                                text: "ADDED:"
                            },
                            D: {
                                color: "#F44336",
                                text: "DELETED:"
                            },
                            A: {
                                color: "#2196F3",
                                text: "ARRAY:"
                            }
                        },
                        R = {
                            level: "log",
                            logger: console,
                            logErrors: !0,
                            collapsed: void 0,
                            predicate: void 0,
                            duration: !1,
                            timestamp: !0,
                            stateTransformer: function (e) {
                                return e
                            },
                            actionTransformer: function (e) {
                                return e
                            },
                            errorTransformer: function (e) {
                                return e
                            },
                            colors: {
                                title: function () {
                                    return "inherit"
                                },
                                prevState: function () {
                                    return "#9E9E9E"
                                },
                                action: function () {
                                    return "#03A9F4"
                                },
                                nextState: function () {
                                    return "#4CAF50"
                                },
                                error: function () {
                                    return "#F20404"
                                }
                            },
                            diff: !1,
                            diffPredicate: void 0,
                            transformer: void 0
                        },
                        q = function () {
                            var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                                t = e.dispatch,
                                n = e.getState;
                            return "function" == typeof t || "function" == typeof n ? A()({
                                dispatch: t,
                                getState: n
                            }) : void console.error("\n[redux-logger v3] BREAKING CHANGE\n[redux-logger v3] Since 3.0.0 redux-logger exports by default logger with default settings.\n[redux-logger v3] Change\n[redux-logger v3] import createLogger from 'redux-logger'\n[redux-logger v3] to\n[redux-logger v3] import { createLogger } from 'redux-logger'\n")
                        };
                    e.defaults = R, e.createLogger = A, e.logger = q, e.default = q, Object.defineProperty(e, "__esModule", {
                        value: !0
                    })
                }(t)
            },
            65: (e, t, n) => {
                "use strict";
                n.r(t), n.d(t, {
                    applyMiddleware: () => D,
                    bindActionCreators: () => S,
                    combineReducers: () => P,
                    compose: () => j,
                    createStore: () => O
                });
                const r = "object" == typeof global && global && global.Object === Object && global;
                var i = "object" == typeof self && self && self.Object === Object && self;
                const o = (r || i || Function("return this")()).Symbol;
                var a = Object.prototype,
                    s = a.hasOwnProperty,
                    u = a.toString,
                    c = o ? o.toStringTag : void 0;
                const l = function (e) {
                    var t = s.call(e, c),
                        n = e[c];
                    try {
                        e[c] = void 0;
                        var r = !0
                    } catch (e) {}
                    var i = u.call(e);
                    return r && (t ? e[c] = n : delete e[c]), i
                };
                var d = Object.prototype.toString;
                const f = function (e) {
                    return d.call(e)
                };
                var p = o ? o.toStringTag : void 0;
                const h = function (e) {
                    return null == e ? void 0 === e ? "[object Undefined]" : "[object Null]" : p && p in Object(e) ? l(e) : f(e)
                };
                const g = function (e, t) {
                    return function (n) {
                        return e(t(n))
                    }
                }(Object.getPrototypeOf, Object);
                const m = function (e) {
                    return null != e && "object" == typeof e
                };
                var v = Function.prototype,
                    y = Object.prototype,
                    b = v.toString,
                    w = y.hasOwnProperty,
                    k = b.call(Object);
                const x = function (e) {
                    if (!m(e) || "[object Object]" != h(e)) return !1;
                    var t = g(e);
                    if (null === t) return !0;
                    var n = w.call(t, "constructor") && t.constructor;
                    return "function" == typeof n && n instanceof n && b.call(n) == k
                };
                var A = n(77),
                    _ = {
                        INIT: "@@redux/INIT"
                    };

                function O(e, t, n) {
                    var r;
                    if ("function" == typeof t && void 0 === n && (n = t, t = void 0), void 0 !== n) {
                        if ("function" != typeof n) throw new Error("Expected the enhancer to be a function.");
                        return n(O)(e, t)
                    }
                    if ("function" != typeof e) throw new Error("Expected the reducer to be a function.");
                    var i = e,
                        o = t,
                        a = [],
                        s = a,
                        u = !1;

                    function c() {
                        s === a && (s = a.slice())
                    }

                    function l() {
                        return o
                    }

                    function d(e) {
                        if ("function" != typeof e) throw new Error("Expected listener to be a function.");
                        var t = !0;
                        return c(), s.push(e),
                            function () {
                                if (t) {
                                    t = !1, c();
                                    var n = s.indexOf(e);
                                    s.splice(n, 1)
                                }
                            }
                    }

                    function f(e) {
                        if (!x(e)) throw new Error("Actions must be plain objects. Use custom middleware for async actions.");
                        if (void 0 === e.type) throw new Error('Actions may not have an undefined "type" property. Have you misspelled a constant?');
                        if (u) throw new Error("Reducers may not dispatch actions.");
                        try {
                            u = !0, o = i(o, e)
                        } finally {
                            u = !1
                        }
                        for (var t = a = s, n = 0; n < t.length; n++) {
                            (0, t[n])()
                        }
                        return e
                    }
                    return f({
                        type: _.INIT
                    }), (r = {
                        dispatch: f,
                        subscribe: d,
                        getState: l,
                        replaceReducer: function (e) {
                            if ("function" != typeof e) throw new Error("Expected the nextReducer to be a function.");
                            i = e, f({
                                type: _.INIT
                            })
                        }
                    })[A.Z] = function () {
                        var e, t = d;
                        return (e = {
                            subscribe: function (e) {
                                if ("object" != typeof e) throw new TypeError("Expected the observer to be an object.");

                                function n() {
                                    e.next && e.next(l())
                                }
                                return n(), {
                                    unsubscribe: t(n)
                                }
                            }
                        })[A.Z] = function () {
                            return this
                        }, e
                    }, r
                }

                function I(e, t) {
                    var n = t && t.type;
                    return "Given action " + (n && '"' + n.toString() + '"' || "an action") + ', reducer "' + e + '" returned undefined. To ignore an action, you must explicitly return the previous state. If you want this reducer to hold no value, you can return null instead of undefined.'
                }

                function P(e) {
                    for (var t = Object.keys(e), n = {}, r = 0; r < t.length; r++) {
                        var i = t[r];
                        0, "function" == typeof e[i] && (n[i] = e[i])
                    }
                    var o = Object.keys(n);
                    var a = void 0;
                    try {
                        ! function (e) {
                            Object.keys(e).forEach((function (t) {
                                var n = e[t];
                                if (void 0 === n(void 0, {
                                        type: _.INIT
                                    })) throw new Error('Reducer "' + t + "\" returned undefined during initialization. If the state passed to the reducer is undefined, you must explicitly return the initial state. The initial state may not be undefined. If you don't want to set a value for this reducer, you can use null instead of undefined.");
                                if (void 0 === n(void 0, {
                                        type: "@@redux/PROBE_UNKNOWN_ACTION_" + Math.random().toString(36).substring(7).split("").join(".")
                                    })) throw new Error('Reducer "' + t + "\" returned undefined when probed with a random type. Don't try to handle " + _.INIT + ' or other actions in "redux/*" namespace. They are considered private. Instead, you must return the current state for any unknown actions, unless it is undefined, in which case you must return the initial state, regardless of the action type. The initial state may not be undefined, but can be null.')
                            }))
                        }(n)
                    } catch (e) {
                        a = e
                    }
                    return function () {
                        var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                            t = arguments[1];
                        if (a) throw a;
                        for (var r = !1, i = {}, s = 0; s < o.length; s++) {
                            var u = o[s],
                                c = n[u],
                                l = e[u],
                                d = c(l, t);
                            if (void 0 === d) {
                                var f = I(u, t);
                                throw new Error(f)
                            }
                            i[u] = d, r = r || d !== l
                        }
                        return r ? i : e
                    }
                }

                function C(e, t) {
                    return function () {
                        return t(e.apply(void 0, arguments))
                    }
                }

                function S(e, t) {
                    if ("function" == typeof e) return C(e, t);
                    if ("object" != typeof e || null === e) throw new Error("bindActionCreators expected an object or a function, instead received " + (null === e ? "null" : typeof e) + '. Did you write "import ActionCreators from" instead of "import * as ActionCreators from"?');
                    for (var n = Object.keys(e), r = {}, i = 0; i < n.length; i++) {
                        var o = n[i],
                            a = e[o];
                        "function" == typeof a && (r[o] = C(a, t))
                    }
                    return r
                }

                function j() {
                    for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                    return 0 === t.length ? function (e) {
                        return e
                    } : 1 === t.length ? t[0] : t.reduce((function (e, t) {
                        return function () {
                            return e(t.apply(void 0, arguments))
                        }
                    }))
                }
                var E = Object.assign || function (e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var n = arguments[t];
                        for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r])
                    }
                    return e
                };

                function D() {
                    for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                    return function (e) {
                        return function (n, r, i) {
                            var o, a = e(n, r, i),
                                s = a.dispatch,
                                u = {
                                    getState: a.getState,
                                    dispatch: function (e) {
                                        return s(e)
                                    }
                                };
                            return o = t.map((function (e) {
                                return e(u)
                            })), s = j.apply(void 0, o)(a.dispatch), E({}, a, {
                                dispatch: s
                            })
                        }
                    }
                }
            },
            77: (e, t, n) => {
                "use strict";
                n.d(t, {
                    Z: () => r
                }), e = n.hmd(e);
                const r = function (e) {
                    var t, n = e.Symbol;
                    return "function" == typeof n ? n.observable ? t = n.observable : (t = n("observable"), n.observable = t) : t = "@@observable", t
                }("undefined" != typeof self ? self : "undefined" != typeof window ? window : void 0 !== n.g ? n.g : e)
            },
            2: function (e, t) {
                var n, r, i;
                r = [e], n = function (e) {
                    "use strict";
                    if ("undefined" == typeof browser) {
                        const t = () => {
                            const e = {
                                alarms: {
                                    clear: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    clearAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    get: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    getAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                bookmarks: {
                                    create: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    export: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    get: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getChildren: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getRecent: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getTree: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    getSubTree: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    import: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    move: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    },
                                    remove: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    removeTree: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    search: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    update: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    }
                                },
                                browserAction: {
                                    getBadgeBackgroundColor: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getBadgeText: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getPopup: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getTitle: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    setIcon: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                commands: {
                                    getAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                contextMenus: {
                                    update: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    },
                                    remove: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    removeAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                cookies: {
                                    get: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getAll: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getAllCookieStores: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    remove: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    set: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                downloads: {
                                    download: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    cancel: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    erase: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getFileIcon: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    open: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    pause: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    removeFile: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    resume: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    search: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    show: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                extension: {
                                    isAllowedFileSchemeAccess: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    isAllowedIncognitoAccess: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                history: {
                                    addUrl: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getVisits: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    deleteAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    deleteRange: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    deleteUrl: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    search: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                i18n: {
                                    detectLanguage: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getAcceptLanguages: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                idle: {
                                    queryState: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                management: {
                                    get: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    getSelf: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    uninstallSelf: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    }
                                },
                                notifications: {
                                    clear: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    create: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    getAll: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    getPermissionLevel: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    update: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    }
                                },
                                pageAction: {
                                    getPopup: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getTitle: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    hide: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    setIcon: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    show: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                runtime: {
                                    getBackgroundPage: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    getBrowserInfo: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    getPlatformInfo: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    openOptionsPage: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    requestUpdateCheck: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    sendMessage: {
                                        minArgs: 1,
                                        maxArgs: 3
                                    },
                                    sendNativeMessage: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    },
                                    setUninstallURL: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                storage: {
                                    local: {
                                        clear: {
                                            minArgs: 0,
                                            maxArgs: 0
                                        },
                                        get: {
                                            minArgs: 0,
                                            maxArgs: 1
                                        },
                                        getBytesInUse: {
                                            minArgs: 0,
                                            maxArgs: 1
                                        },
                                        remove: {
                                            minArgs: 1,
                                            maxArgs: 1
                                        },
                                        set: {
                                            minArgs: 1,
                                            maxArgs: 1
                                        }
                                    },
                                    managed: {
                                        get: {
                                            minArgs: 0,
                                            maxArgs: 1
                                        },
                                        getBytesInUse: {
                                            minArgs: 0,
                                            maxArgs: 1
                                        }
                                    },
                                    sync: {
                                        clear: {
                                            minArgs: 0,
                                            maxArgs: 0
                                        },
                                        get: {
                                            minArgs: 0,
                                            maxArgs: 1
                                        },
                                        getBytesInUse: {
                                            minArgs: 0,
                                            maxArgs: 1
                                        },
                                        remove: {
                                            minArgs: 1,
                                            maxArgs: 1
                                        },
                                        set: {
                                            minArgs: 1,
                                            maxArgs: 1
                                        }
                                    }
                                },
                                tabs: {
                                    create: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    captureVisibleTab: {
                                        minArgs: 0,
                                        maxArgs: 2
                                    },
                                    detectLanguage: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    duplicate: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    executeScript: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    get: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getCurrent: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    },
                                    getZoom: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    getZoomSettings: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    highlight: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    insertCSS: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    move: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    },
                                    reload: {
                                        minArgs: 0,
                                        maxArgs: 2
                                    },
                                    remove: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    query: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    removeCSS: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    sendMessage: {
                                        minArgs: 2,
                                        maxArgs: 3
                                    },
                                    setZoom: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    setZoomSettings: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    update: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    }
                                },
                                webNavigation: {
                                    getAllFrames: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    getFrame: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    }
                                },
                                webRequest: {
                                    handlerBehaviorChanged: {
                                        minArgs: 0,
                                        maxArgs: 0
                                    }
                                },
                                windows: {
                                    create: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    get: {
                                        minArgs: 1,
                                        maxArgs: 2
                                    },
                                    getAll: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    getCurrent: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    getLastFocused: {
                                        minArgs: 0,
                                        maxArgs: 1
                                    },
                                    remove: {
                                        minArgs: 1,
                                        maxArgs: 1
                                    },
                                    update: {
                                        minArgs: 2,
                                        maxArgs: 2
                                    }
                                }
                            };
                            if (0 === Object.keys(e).length) throw new Error("api-metadata.json has not been included in browser-polyfill");
                            class t extends WeakMap {
                                constructor(e, t = void 0) {
                                    super(t), this.createItem = e
                                }
                                get(e) {
                                    return this.has(e) || this.set(e, this.createItem(e)), super.get(e)
                                }
                            }
                            const n = e => e && "object" == typeof e && "function" == typeof e.then,
                                r = e => (...t) => {
                                    chrome.runtime.lastError ? e.reject(chrome.runtime.lastError) : 1 === t.length ? e.resolve(t[0]) : e.resolve(t)
                                },
                                i = (e, t) => {
                                    const n = e => 1 == e ? "argument" : "arguments";
                                    return function (i, ...o) {
                                        if (o.length < t.minArgs) throw new Error(`Expected at least ${t.minArgs} ${n(t.minArgs)} for ${e}(), got ${o.length}`);
                                        if (o.length > t.maxArgs) throw new Error(`Expected at most ${t.maxArgs} ${n(t.maxArgs)} for ${e}(), got ${o.length}`);
                                        return new Promise(((t, n) => {
                                            i[e](...o, r({
                                                resolve: t,
                                                reject: n
                                            }))
                                        }))
                                    }
                                },
                                o = (e, t, n) => new Proxy(t, {
                                    apply: (t, r, i) => n.call(r, e, ...i)
                                });
                            let a = Function.call.bind(Object.prototype.hasOwnProperty);
                            const s = (e, t = {}, n = {}) => {
                                    let r = Object.create(null),
                                        u = {
                                            has: (e, t) => t in e || t in r,
                                            get(e, u, c) {
                                                if (u in r) return r[u];
                                                if (!(u in e)) return;
                                                let l = e[u];
                                                if ("function" == typeof l)
                                                    if ("function" == typeof t[u]) l = o(e, e[u], t[u]);
                                                    else if (a(n, u)) {
                                                    let t = i(u, n[u]);
                                                    l = o(e, e[u], t)
                                                } else l = l.bind(e);
                                                else {
                                                    if ("object" != typeof l || null === l || !a(t, u) && !a(n, u)) return Object.defineProperty(r, u, {
                                                        configurable: !0,
                                                        enumerable: !0,
                                                        get: () => e[u],
                                                        set(t) {
                                                            e[u] = t
                                                        }
                                                    }), l;
                                                    l = s(l, t[u], n[u])
                                                }
                                                return r[u] = l, l
                                            },
                                            set: (e, t, n, i) => (t in r ? r[t] = n : e[t] = n, !0),
                                            defineProperty: (e, t, n) => Reflect.defineProperty(r, t, n),
                                            deleteProperty: (e, t) => Reflect.deleteProperty(r, t)
                                        };
                                    return new Proxy(e, u)
                                },
                                u = {
                                    runtime: {
                                        onMessage: (c = new t((e => "function" != typeof e ? e : function (t, r, i) {
                                            let o = e(t, r);
                                            if (n(o)) return o.then(i, (e => {
                                                console.error(e), i(e)
                                            })), !0;
                                            void 0 !== o && i(o)
                                        })), {
                                            addListener(e, t, ...n) {
                                                e.addListener(c.get(t), ...n)
                                            },
                                            hasListener: (e, t) => e.hasListener(c.get(t)),
                                            removeListener(e, t) {
                                                e.removeListener(c.get(t))
                                            }
                                        })
                                    }
                                };
                            var c;
                            return s(chrome, u, e)
                        };
                        e.exports = t()
                    } else e.exports = browser
                }, void 0 === (i = "function" == typeof n ? n.apply(t, r) : n) || (e.exports = i)
            },
            195: (e, t, n) => {
                "use strict";
                var r = n(0).browser,
                    i = new Map,
                    o = new BroadcastChannel("workerfactory-inner");
                o.addEventListener("message", (function (e) {
                    if ("spawn-worker" == e.data.type) {
                        var t = e.data.path;
                        if (i.has(t)) throw new Error("Worker already running: " + t);
                        var n = r.runtime.getURL("/content/" + t),
                            a = new Worker(n);
                        i.set(t, a)
                    } else if ("kill-worker" == e.data.type) {
                        var s = e.data.path,
                            u = i.get(s);
                        if (!u) throw new Error("Worker not running.");
                        u.terminate(), i.delete(s), o.postMessage({
                            type: "worker-killed",
                            worker_path: s
                        })
                    }
                }))
            },
            189: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = function e(t, n, r) {
                        null === t && (t = Function.prototype);
                        var i = Object.getOwnPropertyDescriptor(t, n);
                        if (void 0 === i) {
                            var o = Object.getPrototypeOf(t);
                            return null === o ? void 0 : e(o, n, r)
                        }
                        if ("value" in i) return i.value;
                        var a = i.get;
                        return void 0 !== a ? a.call(r) : void 0
                    },
                    i = function () {
                        function e(e, t) {
                            for (var n = 0; n < t.length; n++) {
                                var r = t[n];
                                r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                            }
                        }
                        return function (t, n, r) {
                            return n && e(t.prototype, n), r && e(t, r), t
                        }
                    }();

                function o(e, t) {
                    if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                    return !t || "object" != typeof t && "function" != typeof t ? e : t
                }

                function a(e, t) {
                    if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                    e.prototype = Object.create(t && t.prototype, {
                        constructor: {
                            value: e,
                            enumerable: !1,
                            writable: !0,
                            configurable: !0
                        }
                    }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                }

                function s(e, t) {
                    if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                }
                t.describeAll = function () {
                    var e = {};
                    return J((function (t) {
                        e[t.name] = {
                            name: t.name,
                            title: t.title,
                            description: t.description,
                            icon: t.icon,
                            icon18: t.icon.replace(/\-(\d+)\./, "-$1."),
                            catPriority: t.catPriority || 0
                        }
                    })), e
                }, t.register = X, t.availableActions = function (e) {
                    var t = [];
                    J((function (n) {
                        n.canPerform(e) && t.push(n.name)
                    }));
                    var n = [u.prefs["default-action-0"], u.prefs["default-action-1"], u.prefs["default-action-2"]];
                    return t.sort((function (e, t) {
                        var r = Y[e],
                            i = Y[t],
                            o = r.catPriority || 0,
                            a = i.catPriority || 0;
                        return a != o ? a - o : e == n[o] ? -1 : t == n[a] ? 1 : i.priority - r.priority
                    })), t
                }, t.execute = function (e, t) {
                    var n = Y[e];
                    if (!n) throw new Error("No such action " + e);
                    return new n(t).execute(), n.keepOpen
                }, t.mergeLocal = $, t.convertLocal = ee;
                var u = n(3),
                    c = u.browser,
                    l = n(1),
                    d = n(50),
                    f = n(190),
                    p = n(12),
                    h = n(7),
                    g = n(194),
                    m = n(34),
                    v = n(191),
                    y = n(238),
                    b = n(192),
                    w = n(217),
                    k = n(218),
                    x = n(243),
                    A = n(79),
                    _ = n(197),
                    O = n(196),
                    I = n(8),
                    P = I.buildOptions.noyt || !1,
                    C = n(223),
                    S = h.Concurrent((function () {
                        return u.prefs.downloadControlledMax || 1 / 0
                    })),
                    j = h.Concurrent((function () {
                        return u.prefs.convertControlledMax || 1 / 0
                    })),
                    E = {},
                    D = {},
                    T = 0,
                    R = t.Action = function () {
                        function e(t) {
                            s(this, e), this.hit = Object.assign({}, t), this.reqs = {}, this.actionId = ++T, this.cleanupData = {
                                files: []
                            }
                        }
                        return i(e, [{
                            key: "execute",
                            value: function () {
                                var e = this;
                                e.updateRunning(1), Promise.resolve(e.getReqs()).then((function () {
                                    return e.solveAllReqs()
                                })).then((function () {
                                    return Promise.resolve(e.doJob())
                                })).then((function () {
                                    return Promise.resolve(e.postJob())
                                })).catch((function (t) {
                                    console.warn("Action error:", t.message), t.noReport || (!t.videoTitle && e.hit.title && (t.videoTitle = e.hit.title), t.reportAsLog ? f.log(t) : f.error(t))
                                })).then((function () {
                                    return Promise.resolve(e.cleanup())
                                })).then((function () {
                                    e.updateRunning(-1), e.setOperation(null)
                                }))
                            }
                        }, {
                            key: "getReqs",
                            value: function () {}
                        }, {
                            key: "solveReqs",
                            value: function () {}
                        }, {
                            key: "solveAllReqs",
                            value: function () {
                                var e = this;
                                return Promise.resolve(e.solveReqs()).then((function (t) {
                                    if (t) return e.solveAllReqs()
                                }))
                            }
                        }, {
                            key: "solveCoAppReqs",
                            value: function () {
                                var e = this;
                                return new Promise((function (t, n) {
                                    m.check().then((function (r) {
                                        if (delete e.reqs.coapp, r.status) {
                                            e.hasCoapp = !0;
                                            var i = r.info.version;
                                            e.reqs.coappMin && !h.isMinimumVersion(r.info.version, e.reqs.coappMin) ? m.call("quit").catch((function () {})).then((function () {
                                                return new Promise((function (e, t) {
                                                    setTimeout((function () {
                                                        e()
                                                    }), u.prefs.coappRestartDelay)
                                                }))
                                            })).then((function () {
                                                m.check().then((function (r) {
                                                    r.status && h.isMinimumVersion(r.info.version, e.reqs.coappMin) ? (delete e.reqs.coappMin, t(!0)) : (g.alert({
                                                        title: u._("coapp_outofdate"),
                                                        text: u._("coapp_outofdate_text", [r.info && r.info.version || i, e.reqs.coappMin]),
                                                        buttons: [{
                                                            text: u._("coapp_update"),
                                                            className: "btn-success",
                                                            rpcMethod: "installCoApp"
                                                        }]
                                                    }), delete e.reqs.coappMin, n(new h.VDHError("Aborted", {
                                                        noReport: !0
                                                    })))
                                                }))
                                            })) : (delete e.reqs.coappMin, t(!0))
                                        } else g.alert({
                                            title: u._("coapp_required"),
                                            text: u._("coapp_required_text"),
                                            buttons: [{
                                                text: u._("coapp_install"),
                                                className: "btn-success",
                                                rpcMethod: "installCoApp"
                                            }]
                                        }), n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    }))
                                }))
                            }
                        }, {
                            key: "doJob",
                            value: function () {
                                console.warn("Generic action doJob")
                            }
                        }, {
                            key: "postJob",
                            value: function () {}
                        }, {
                            key: "cleanup",
                            value: function () {}
                        }, {
                            key: "start",
                            value: function () {
                                return console.warn("action.start() is obsolete, use action.execute()"), this.execute()
                            }
                        }, {
                            key: "setOperation",
                            value: function (e) {
                                d.setHitOperation(this.hit.id, e)
                            }
                        }, {
                            key: "setProgress",
                            value: function (e) {
                                d.updateProgress(this.hit.id, e)
                            }
                        }, {
                            key: "clearProgress",
                            value: function () {
                                d.updateProgress(this.hit.id, null)
                            }
                        }, {
                            key: "updateRunning",
                            value: function (e) {
                                d.updateRunning(this.hit.id, e)
                            }
                        }, {
                            key: "updateHit",
                            value: function (e) {
                                d.update(this.hit.id, e)
                            }
                        }, {
                            key: "setAbort",
                            value: function (e) {
                                var t = E[this.hit.id];
                                t || (t = E[this.hit.id] = {}), t[this.actionId] && console.warn("Overwritting abortable task"), t[this.actionId] = e.bind(this), this.updateHit({})
                            }
                        }, {
                            key: "clearAbort",
                            value: function () {
                                var e = E[this.hit.id];
                                e && (e[this.actionId] && (delete e[this.actionId], 0 == Object.keys(e).length && delete E[this.hit.id]))
                            }
                        }, {
                            key: "setStop",
                            value: function (e) {
                                var t = D[this.hit.id];
                                t || (t = D[this.hit.id] = {}), t[this.actionId] && console.warn("Overwritting stoppable task"), t[this.actionId] = e.bind(this), this.updateHit({})
                            }
                        }, {
                            key: "clearStop",
                            value: function () {
                                var e = D[this.hit.id];
                                e && (e[this.actionId] && (delete e[this.actionId], 0 == Object.keys(e).length && delete D[this.hit.id]))
                            }
                        }], [{
                            key: "keepOpen",
                            get: function () {
                                return !1
                            }
                        }]), e
                    }(),
                    q = t.DownloadAction = function (e) {
                        function t(e) {
                            s(this, t);
                            var n = o(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                            return n.promptFilename = !0, n.streams = {}, n.hasCoapp = !1, n
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                var e = this;
                                return this.getStreams(), e.solveAllStreamsName().then((function () {
                                    return e.ensureOutputDirectory()
                                })).then((function () {
                                    return e.downloadAllStreams()
                                })).then((function () {
                                    return e.grabInfo()
                                })).then((function () {
                                    return e.handleWatermark()
                                })).then((function () {
                                    if (e.reqs.aggregate) return e.aggregate()
                                })).then((function () {
                                    if (e.reqs.convert) return e.convert()
                                })).then((function () {
                                    x.newDownload()
                                })).then((function () {
                                    e.watermarked && e.explainQR()
                                })).then((function () {
                                    C.downloadSuccess(e.hit)
                                })).catch((function (t) {
                                    throw C.downloadError(e.hit, t.message), t
                                }))
                            }
                        }, {
                            key: "getCoappTmpName",
                            value: function () {
                                var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {};
                                return m.call("tmp.tmpName", {
                                    prefix: e.prefix || "vdh-",
                                    postfix: e.postfix || ".tmp"
                                })
                            }
                        }, {
                            key: "solveStreamName",
                            value: function (e) {
                                var t = this;
                                if (t.reqs.aggregate || t.reqs.convert) return t.getCoappTmpName().then((function (n) {
                                    e.fileName = n.fileName, e.directory = n.directory, e.filePath = n.filePath, t.cleanupData.files.push(n.filePath), t.cleanupData.files.push(n.filePath + ".part")
                                }));
                                if ("coapp" == t.downloadWith) {
                                    if (t.hit.urls) return new Promise((function (n, r) {
                                        var i = void 0,
                                            o = t.hit.type,
                                            a = /([^/]*)\.([a-zA-Z0-9]+)(?:\?.*)?$/.exec(e.url);
                                        a ? (i = a[2], o = a[1]) : i = Object.keys(t.hit.extensions)[0];
                                        var s = ("000000" + e.index).substr(-6);
                                        switch (u.prefs.galleryNaming) {
                                            case "type-index":
                                                e.fileName = A.getFilenameFromTitle(t.hit.type + "-" + s, i);
                                                break;
                                            case "url":
                                                e.fileName = A.getFilenameFromTitle(o, i);
                                                break;
                                            case "index-url":
                                                e.fileName = A.getFilenameFromTitle(s + "-" + o, i)
                                        }
                                        e.directory = t.filePath, m.call("path.homeJoin", e.directory, e.fileName).then((function (t) {
                                            e.filePath = t, n()
                                        })).catch(r)
                                    }));
                                    e.fileName = t.fileName, e.directory = t.directory, e.filePath = t.filePath
                                } else e.fileName = t.getFilename(), t.reqs.needFilename && (e.saveas = !0)
                            }
                        }, {
                            key: "solveAllStreamsName",
                            value: function () {
                                var e = this;
                                return Promise.all(Object.keys(this.streams).map((function (t) {
                                    return e.solveStreamName(e.streams[t])
                                })))
                            }
                        }, {
                            key: "downloadAllStreams",
                            value: function () {
                                var e = this,
                                    t = this;
                                return t.setOperation("queued"), S((function () {
                                    return t.clearAbort(), t.updateHit({
                                        operation: "downloading",
                                        opStartDate: Date.now()
                                    }), t.hit.chunked ? t.downloadAllChunkedStreams() : (t.setAbort(t.abortDownload), t.downloadStreamControl = h.Concurrent((function () {
                                        return u.prefs.downloadStreamControlledMax || 1 / 0
                                    })), Promise.all(Object.keys(e.streams).map((function (e) {
                                        return t.downloadStream(t.streams[e])
                                    }))))
                                }), (function (e, n) {
                                    t.setAbort((function () {
                                        n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    }))
                                })).then((function () {
                                    t.clearAbort()
                                })).catch((function (e) {
                                    throw t.clearAbort(), e
                                }))
                            }
                        }, {
                            key: "downloadStream",
                            value: function (e) {
                                var t = this;
                                return t.downloadStreamControl((function () {
                                    return t.doDownloadStream(e)
                                }))
                            }
                        }, {
                            key: "doDownloadStream",
                            value: function (e) {
                                var t = this,
                                    n = {
                                        coappDownload: "coapp" === t.downloadWith,
                                        proxy: this.hit.proxy,
                                        source: {
                                            url: e.url,
                                            isPrivate: this.hit.isPrivate,
                                            headers: e.headers,
                                            referrer: e.referrer
                                        },
                                        target: {
                                            filename: e.fileName || this.getFilename(),
                                            directory: e.directory,
                                            saveAs: !!e.saveas
                                        }
                                    };
                                return new Promise((function (r, i) {
                                    e.downloadId = y.download(n, (function (n) {
                                        n && !t.filePath && (t.filePath = n), n && u.prefs.contentRedirectEnabled && t.hit.possibleContentRedirect && (void 0 === e.contentRedirected && (e.contentRedirected = 3), e.contentRedirect > 0) ? m.call("fs.stat", n).then((function (o) {
                                            o.size < 2048 ? m.call("fs.readFile", n, "utf8").then((function (n) {
                                                f.log({
                                                    message: "Short content " + o.size,
                                                    details: n
                                                });
                                                var a = /^(https?:\/\/.*)/.exec(n) || /^(https?:\/\/.*)/.exec(n.substring(6)),
                                                    s = a && a[1];
                                                s ? (e.contentRedirect--, e.url = s, t.downloadStream(e).then((function (e) {
                                                    r(e)
                                                })).catch((function (e) {
                                                    i(e)
                                                }))) : r()
                                            })).catch((function (e) {
                                                i(e)
                                            })) : r()
                                        })).catch((function (e) {
                                            i(e)
                                        })) : r()
                                    }), (function (e) {
                                        i(e)
                                    }), (function (n) {
                                        e.progress = n, n = 0, Object.keys(t.streams).forEach((function (e) {
                                            n += t.streams[e].progress || 0
                                        })), n /= Object.keys(t.streams).length, t.setProgress(n)
                                    }), t.hit.headers)
                                }))
                            }
                        }, {
                            key: "downloadAllChunkedStreams",
                            value: function () {
                                var e = this,
                                    t = [];
                                Object.keys(e.streams).forEach((function (n) {
                                    var r = e.streams[n];
                                    t.push({
                                        fileName: r.filePath,
                                        type: n
                                    })
                                }));
                                var n = "hls" == e.hit.chunked;
                                n && e.setStop(e.stopRecording);
                                var r = "dash-adp" == e.hit.chunked;
                                return r && e.setAbort(e.abortChunkedDownload), new Promise((function (i, o) {
                                    w.download(e, t, (function () {
                                        n && e.clearStop(), r && e.clearAbort(), i()
                                    }), (function (t) {
                                        n && e.clearStop(), r && e.clearAbort(), o(t)
                                    }), (function (t) {
                                        e.setProgress(t)
                                    }))
                                }))
                            }
                        }, {
                            key: "ensureOutputDirectory",
                            value: function () {
                                var e = this;
                                if (e.directory) return m.call("fs.stat", e.directory).then((function (t) {
                                    if (!(16384 & t.mode)) throw new h.DetailsError(u._("error_not_directory", e.directory), "details")
                                }), (function (t) {
                                    return new Promise((function (t, n) {
                                        m.call("fs.mkdirp", e.directory).then((function () {
                                            t()
                                        }))
                                    }))
                                }))
                            }
                        }, {
                            key: "grabInfo",
                            value: function () {
                                var e = this,
                                    t = e.streams.video || e.streams.full || e.streams.audio;
                                if (t && t.filePath) return b.info(t.filePath).then((function (t) {
                                    var n = {};
                                    t.width && t.height && (e.videoInfo = t, n.size = t.width + "x" + t.height), t.duration && (n.duration = t.duration), t.fps && (n.fps = t.fps), Object.assign(e.hit, n), e.updateHit(n)
                                })).catch((function (n) {
                                    var r = {
                                        file: t.filePath
                                    };
                                    m.call("fs.stat", t.filePath).then((function (e) {
                                        r.stat = e
                                    })).catch((function (e) {
                                        r.error = e.message
                                    })).then((function () {
                                        f.log({
                                            message: u._("corrupted_media_file", [e.hit.title, t.filePath]),
                                            details: JSON.stringify(r, null, 4) + "\n" + n.message
                                        })
                                    }))
                                }))
                            }
                        }, {
                            key: "handleWatermark",
                            value: function () {
                                var e = this;
                                if (e.videoInfo && e.license && "accepted" != e.license.status && "unneeded" != e.license.status) return b.wmForHeight(e.videoInfo.height).then((function (t) {
                                    e.watermark = t, t && t.qr && (e.cleanupData.files.push(t.qr), e.watermarked = !0)
                                }))
                            }
                        }, {
                            key: "aggregate",
                            value: function () {
                                var e = this;
                                return e.setOperation("converter_queued"), e.setProgress(0), j((function () {
                                    return e.doAggregate()
                                }), (function (t, n) {
                                    e.setAbort((function () {
                                        n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    }))
                                })).then((function () {
                                    e.clearAbort()
                                })).catch((function (t) {
                                    throw e.clearAbort(), t
                                }))
                            }
                        }, {
                            key: "doAggregate",
                            value: function () {
                                var e = this;
                                return e.updateHit({
                                    operation: "aggregating",
                                    opStartDate: Date.now()
                                }), e.setAbort((function () {
                                    e.aborted = !0
                                })), new Promise((function (t, n) {
                                    b.aggregate({
                                        audio: e.streams.audio.filePath,
                                        video: e.streams.video.filePath,
                                        target: e.intermediateFilePath || e.filePath,
                                        wm: e.watermark,
                                        videoCodec: e.videoInfo && e.videoInfo.videoCodec,
                                        fps: e.videoInfo && e.videoInfo.fps
                                    }, (function (t) {
                                        if (e.aborted) throw new Error("Aborted");
                                        var n = e.videoInfo && e.videoInfo.duration || e.hit.duration;
                                        if (n) {
                                            var r = Math.round(100 * t / n);
                                            e.setProgress(r)
                                        }
                                    })).then(t).catch((function (t) {
                                        n(new h.DetailsError(u._("failed_aggregating", e.hit.title), t.details || t.message || ""))
                                    }))
                                })).then((function () {
                                    e.clearAbort(), delete e.watermark
                                })).catch((function (t) {
                                    throw e.clearAbort(), t
                                }))
                            }
                        }, {
                            key: "convert",
                            value: function () {
                                var e = this;
                                return e.setOperation("converter_queued"), e.setProgress(0), j((function () {
                                    return e.doConvert()
                                }), (function (t, n) {
                                    e.setAbort((function () {
                                        n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    }))
                                })).then((function () {
                                    e.clearAbort()
                                })).catch((function (t) {
                                    throw e.clearAbort(), t
                                }))
                            }
                        }, {
                            key: "doConvert",
                            value: function () {
                                var e = this;
                                return e.updateHit({
                                    operation: "converting",
                                    opStartDate: Date.now()
                                }), e.setAbort((function () {
                                    e.aborted = !0
                                })), new Promise((function (t, n) {
                                    var r = e.intermediateFilePath || e.reqs.convert.audioonly && e.streams.audio && e.streams.audio.filePath || e.streams.full && e.streams.full.filePath || e.streams.video && e.streams.video.filePath || e.streams.audio && e.streams.video.filePath;
                                    if (!r) return n(new Error("Could not determine conversion source"));
                                    b.convert({
                                        source: r,
                                        target: e.filePath,
                                        config: e.reqs.convert,
                                        wm: e.watermark
                                    }, (function (t) {
                                        if (e.aborted) throw new Error("Aborted");
                                        var n = e.videoInfo && e.videoInfo.duration || e.hit.duration;
                                        if (n) {
                                            var r = Math.round(100 * t / n);
                                            e.setProgress(r)
                                        }
                                    })).then(t).catch((function (t) {
                                        n(new h.DetailsError(u._("failed_converting", e.hit.title), t.details || t.message || ""))
                                    }))
                                })).then((function () {
                                    e.clearAbort()
                                })).catch((function (t) {
                                    throw e.clearAbort(), t
                                }))
                            }
                        }, {
                            key: "getStreams",
                            value: function () {
                                var e = this;
                                this.hit.url ? this.streams.full = {
                                    url: this.hit.url
                                } : this.hit.urls ? this.hit.urls.forEach((function (t, n) {
                                    e.streams["doc" + n] = {
                                        url: new URL(t, e.hit.baseUrl).href,
                                        index: n
                                    }
                                })) : (this.hit.audioUrl && (this.streams.audio = {
                                    url: this.hit.audioUrl,
                                    contentRedirect: 2
                                }), !this.hit.videoUrl || this.reqs.convert && this.reqs.convert.audioonly || (this.streams.video = {
                                    url: this.hit.videoUrl,
                                    contentRedirect: 2
                                })), Object.keys(this.streams).forEach((function (t) {
                                    var n = e.streams[t];
                                    n.type = t, n.headers = e.hit.headers, n.referrer = e.hit.referrer
                                }))
                            }
                        }, {
                            key: "getReqs",
                            value: function () {
                                var e = this;
                                return P && O.matchHit(this.hit) ? (O.forbidden(), Promise.reject(new h.VDHError("Forbidden", {
                                    noReport: !0
                                }))) : Promise.resolve(r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "getReqs", this).call(this)).then((function () {
                                    if (!e.reqs.convert) return _.outputConfigForHit(e.hit).then((function (t) {
                                        t && (e.hit.extension = t.ext || t.params.f || e.hit.extension), e.reqs.convert = t
                                    }))
                                })).then((function () {
                                    e.promptFilename && (e.reqs.needFilename = !0), e.hit.convert && (e.reqs.convert = e.hit.convert), e.hit.urls || e.hit.audioUrl && e.hit.videoUrl || e.reqs.convert || e.hit.chunked || "coapp" == u.prefs.coappDownloads ? (e.reqs.coappMin = "1.2.1", e.downloadWith = "coapp", e.reqs.promptFilename = e.reqs.needFilename) : "ask" == u.prefs.coappDownloads && (e.reqs.askDownloadMode = !0), !e.hit.audioUrl || !e.hit.videoUrl || e.reqs.convert && e.reqs.convert.audioonly || (e.reqs.aggregate = !0), e.hit.chunked ? (e.reqs.coapp = !0, e.reqs.needFilename && (e.reqs.promptFilename = !0), "audio-video" == u.prefs.dashOnAdp && (u.prefs.dashOnAdp = "audio_video"), "audio_video" != u.prefs.dashOnAdp && delete e.reqs.aggregate, "hls" == e.hit.chunked && (t.hlsDownloadingCount++, e.hlsDownloadingCounted = !0, v.hlsDownloadLimit && (e.reqs.checkHlsDownloadLimit = !0))) : (e.hit.urls || e.hit.audioUrl && e.hit.videoUrl) && (e.reqs.coapp = !0), e.reqs.convert && e.reqs.convert.audioonly && (e.reqs.license = !0), (e.reqs.coappMin || e.reqs.aggregate || e.reqs.convert || e.reqs.license) && (e.reqs.coapp = !0), e.hit.urls && (delete e.reqs.convert, delete e.reqs.aggregate), (e.reqs.aggregate || e.reqs.convert || e.reqs.license) && (e.reqs.checkLicense = !0), e.reqs.aggregate && e.reqs.convert && (e.reqs.intermediateFilePath = !0), e.reqs.coappMin && e.hit.proxy && "http" == e.hit.proxy.type.substr(0, 4) && (e.reqs.coappMin = "1.2.1"), e.reqs.coappMin && e.hit.possibleContentRedirect && (e.reqs.coappMin = "1.2.1"), e.reqs.coappMin && e.hit.urls && (e.reqs.coappMin = "1.2.2"), "chrome" == I.buildOptions.browser && e.reqs.coappMin && !h.isMinimumVersion(e.reqs.coappMin, "1.2.3") && (e.reqs.coappMin = "1.2.3"), "edge" == I.buildOptions.browser && e.reqs.coappMin && !h.isMinimumVersion(e.reqs.coappMin, "1.6.1") && (e.reqs.coappMin = "1.6.1"), e.reqs.coappMin && u.prefs.forcedCoappVersion && (e.reqs.coappMin = u.prefs.forcedCoappVersion)
                                }))
                            }
                        }, {
                            key: "solveReqs",
                            value: function () {
                                var e = this;
                                if (e.reqs.askDownloadMode) return new Promise((function (t, n) {
                                    g.alert({
                                        title: u._("download_method"),
                                        text: [u._("download_modes1"), u._("download_modes2")],
                                        height: 350,
                                        buttons: [{
                                            text: u._("download_with_browser"),
                                            className: "btn-primary",
                                            close: !0,
                                            trigger: {
                                                mode: "browser"
                                            }
                                        }, {
                                            text: u._("download_with_coapp"),
                                            className: "btn-success",
                                            close: !0,
                                            trigger: {
                                                mode: "coapp"
                                            }
                                        }],
                                        notAgain: u._("download_method_not_again")
                                    }).then((function (n) {
                                        e.downloadWith = n.mode, n.notAgain && (u.prefs.coappDownloads = n.mode), "coapp" == n.mode && (e.reqs.coappMin = "1.2.1", e.reqs.coapp = !0, e.reqs.needFilename && (e.reqs.promptFilename = !0)), delete e.reqs.askDownloadMode, t(!0)
                                    })).catch((function () {
                                        n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    }))
                                }));
                                if (e.reqs.coapp) return e.solveCoAppReqs();
                                if (e.reqs.checkLicense) return new Promise((function (t, n) {
                                    v.checkLicense().then((function (r) {
                                        delete e.reqs.checkLicense, e.license = r, e.reqs.license && "accepted" != r.status && "unneeded" != r.status ? (delete e.reqs.license, v.alertAudioNeedsReg(), n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))) : t(!0)
                                    }))
                                }));
                                if (e.reqs.checkHlsDownloadLimit) {
                                    var n = (u.prefs.lastHlsDownload || 0) + 60 * v.hlsDownloadLimit * 1e3;
                                    if (t.hlsDownloadingCount > 1 || Date.now() < n) return new Promise((function (t, n) {
                                        v.checkLicense().then((function (r) {
                                            delete e.reqs.checkHlsDownloadLimit, e.license = r, "accepted" != r.status && "unneeded" != r.status ? (delete e.reqs.license, v.alertHlsDownloadLimit(), n(new h.VDHError("Aborted", {
                                                noReport: !0
                                            }))) : t(!0)
                                        }))
                                    }))
                                }
                                return e.reqs.promptFilename ? new Promise((function (t, n) {
                                    delete e.reqs.promptFilename, delete e.reqs.needFilename;
                                    var r = e.getFilename({
                                        noExtension: !!e.hit.urls
                                    });
                                    g.saveAs(r, u.prefs.lastDownloadDirectory || "dwhelper", {
                                        dirOnly: !!e.hit.urls
                                    }).then((function (r) {
                                        r ? (u.prefs.rememberLastDir && (u.prefs.lastDownloadDirectory = r.directory), e.filePath = r.filePath, e.directory = r.directory, e.fileName = r.fileName, t(!0)) : n(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    })).catch(n)
                                })) : e.fileName || "coapp" != e.downloadWith ? e.reqs.intermediateFilePath ? new Promise((function (t, n) {
                                    delete e.reqs.intermediateFilePath, e.getCoappTmpName({
                                        postfix: "." + (e.hit.originalExt || e.hit.extension)
                                    }).then((function (n) {
                                        e.intermediateFilePath = n.filePath, e.cleanupData.files.push(n.filePath), t(!0)
                                    })).catch(n)
                                })) : void 0 : new Promise((function (t, n) {
                                    delete e.reqs.needFilename;
                                    var r = e.getFilename({
                                        noExtension: !!e.hit.urls
                                    });
                                    m.call("makeUniqueFileName", u.prefs.lastDownloadDirectory || "dwhelper", r).then((function (n) {
                                        e.filePath = n.filePath, e.directory = n.directory, e.fileName = n.fileName, t(!0)
                                    })).catch(n)
                                }))
                            }
                        }, {
                            key: "postJob",
                            value: function () {
                                var e = this;
                                return Promise.resolve(r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "postJob", this).call(this)).then((function () {
                                    u.prefs.autoPin && e.updateHit({
                                        status: "pinned",
                                        pinned: !0
                                    })
                                })).then((function () {
                                    e.filePath && e.updateHit({
                                        localFilePath: e.filePath,
                                        localDirectory: e.directory || void 0
                                    })
                                })).then((function () {
                                    u.prefs.notifyReady && (e.hit.isPrivate && u.prefs.noPrivateNotification || c.notifications.create(e.hit.id, {
                                        type: "basic",
                                        title: u._("vdh_notification"),
                                        message: u._("file_ready", e.filePath || e.hit.title || u._("media")),
                                        iconUrl: c.runtime.getURL("content/images/icon-36.png")
                                    }))
                                })).then((function () {
                                    e.hlsDownloadingCounted && (u.prefs.lastHlsDownload = Date.now())
                                }))
                            }
                        }, {
                            key: "getFilename",
                            value: function (e) {
                                e = e || {};
                                var t = this.hit.title || "video",
                                    n = null;
                                return e.noExtension || (n = this.hit.extension || "mp4"), A.getFilenameFromTitle(t, n)
                            }
                        }, {
                            key: "getTmpFileName",
                            value: function () {
                                return "vdh-" + Math.round(1e9 * Math.random()) + ".tmp"
                            }
                        }, {
                            key: "abortDownload",
                            value: function () {
                                var e = this;
                                Object.keys(e.streams).forEach((function (t) {
                                    var n = e.streams[t];
                                    n.downloadId && y.abort(n.downloadId)
                                }))
                            }
                        }, {
                            key: "abortChunkedDownload",
                            value: function () {
                                this.abortChunked ? this.abortChunked() : console.warn("trying to abort chunked download but not abort function")
                            }
                        }, {
                            key: "stopRecording",
                            value: function () {
                                "hls" == this.hit.chunked && k.stopRecording(this.hit.id)
                            }
                        }, {
                            key: "cleanup",
                            value: function () {
                                var e = this;
                                return e.hlsDownloadingCounted && t.hlsDownloadingCount--, Promise.resolve(r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "cleanup", this).call(this)).then((function () {
                                    return e.clearProgress(), e.hasCoapp && !u.prefs.converterKeepTmpFiles && e.cleanupData.files.forEach((function (e) {
                                        m.call("fs.unlink", e).catch((function () {}))
                                    })), Promise.resolve()
                                }))
                            }
                        }, {
                            key: "explainQR",
                            value: function () {
                                u.prefs.qrMessageNotAgain || u.ui.open("explainqr#" + encodeURIComponent(this.hit.id), {
                                    type: u.prefs.alertDialogType,
                                    url: "content/explain-qr.html"
                                })
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !(e.running > 0) && (void 0 !== e.url || void 0 !== e.audioUrl || void 0 !== e.videoUrl || void 0 !== e.urls)
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "download"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_download_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_download_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-download-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 100
                            }
                        }]), t
                    }(R);
                q.hlsDownloadingCount = 0;
                var M = function (e) {
                        function t(e) {
                            s(this, t);
                            var n = o(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                            return n.promptFilename = !1, n
                        }
                        return a(t, e), i(t, null, [{
                            key: "canPerform",
                            value: function (e) {
                                return !(e.running > 0) && (void 0 !== e.url || void 0 !== e.audioUrl || void 0 !== e.videoUrl || void 0 !== e.urls)
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "quickdownload"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_quickdownload_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_quickdownload_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-quick-download2-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 90
                            }
                        }]), t
                    }(q),
                    F = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                var e = E[this.hit.id] || {};
                                Object.keys(e).forEach((function (t) {
                                    e[t](), delete e[t]
                                })), 0 == Object.keys(e).length && delete E[this.hit.id]
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !!E[e.id]
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "abort"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_abort_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_abort_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-abort-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 300
                            }
                        }, {
                            key: "catPriority",
                            get: function () {
                                return 2
                            }
                        }]), t
                    }(R),
                    W = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                var e = D[this.hit.id] || {};
                                Object.keys(e).forEach((function (t) {
                                    e[t](), delete e[t]
                                })), 0 == Object.keys(e).length && delete D[this.hit.id]
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !!D[e.id] && !("hls" != e.chunked || !k.canStop(e.id))
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "stop"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_stop_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_stop_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-stoprecord-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 300
                            }
                        }, {
                            key: "catPriority",
                            get: function () {
                                return 2
                            }
                        }]), t
                    }(R),
                    N = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "getReqs",
                            value: function () {
                                var e = this,
                                    n = this;
                                return P && O.matchHit(this.hit) ? (O.forbidden(), u.ui.close("main"), Promise.reject(new h.VDHError("Forbidden", {
                                    noReport: !0
                                }))) : Promise.resolve().then((function () {
                                    var t = "dlconv#" + e.hit.id;
                                    return u.openedContents().indexOf("main") >= 0 ? u.rpc.call("main", "embed", c.runtime.getURL("content/dlconv-embed.html?panel=" + t)).then((function () {
                                        return u.wait(t)
                                    })).catch((function (e) {
                                        throw new h.VDHError("Aborted", {
                                            noReport: !0
                                        })
                                    })) : b.getOutputConfigs().then((function (e) {
                                        var t = u.prefs.dlconvLastOutput || "05cb6b27-9167-4d83-833d-218a107d0376",
                                            n = e[t];
                                        if (!n) throw new Error("No such output configuration");
                                        return {
                                            outputConfigId: t,
                                            outputConfig: n
                                        }
                                    }))
                                })).then((function (e) {
                                    u.prefs.dlconvLastOutput = e.outputConfigId;
                                    var t = e.outputConfig;
                                    n.hit.extension && (n.hit.originalExt = n.hit.extension), n.hit.extension = t.ext || t.params.f || n.hit.extension, n.reqs.convert = e.outputConfig, n.promptFilename = e.prompt
                                })).then((function () {
                                    return r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "getReqs", e).call(e)
                                }))
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !(e.running > 0) && (void 0 !== e.url || void 0 !== e.audioUrl || void 0 !== e.videoUrl)
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "downloadconvert"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_downloadconvert_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_downloadconvert_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-download-convert-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 80
                            }
                        }, {
                            key: "keepOpen",
                            get: function () {
                                return !0
                            }
                        }]), t
                    }(q),
                    U = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return u.ui.open("details#" + encodeURIComponent(this.hit.id), {
                                    type: "tab",
                                    url: "content/details.html"
                                }), Promise.resolve()
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !0
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "details"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_details_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_details_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-details-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 0
                            }
                        }]), t
                    }(R),
                    z = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return l.call("main", "copyToClipboard", this.hit.url)
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return void 0 !== e.url
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "copyurl"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_copyurl_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_copyurl_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-copy-link-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 30
                            }
                        }]), t
                    }(R),
                    L = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return p.dispatch("hit.delete", this.hit.id), Promise.resolve()
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !0
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "deletehit"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_deletehit_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_deletehit_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-delete-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 0
                            }
                        }]), t
                    }(R),
                    V = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return this.updateHit({
                                    pinned: !0
                                }), Promise.resolve()
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !e.pinned
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "pin"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_pin_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_pin_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-pin-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 0
                            }
                        }]), t
                    }(R),
                    H = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return b.play(this.hit.localFilePath).catch((function (e) {
                                    throw new h.DetailsError(u._("failed_playing_file"), e.message)
                                }))
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !e.urls && 0 === e.running && u.prefs.avplayEnabled && !!e.localFilePath
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "cvplay"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_avplay_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_avplay_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-avplay-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 180
                            }
                        }, {
                            key: "catPriority",
                            get: function () {
                                return 1
                            }
                        }]), t
                    }(R),
                    K = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return l.call("main", "embed", c.runtime.getURL("content/blacklist-embed.html?panel=blacklist#" + encodeURIComponent(this.hit.id))), Promise.resolve()
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return void 0 !== e.url || void 0 !== e.audioUrl || void 0 !== e.videoUrl || void 0 !== e.topUrl || void 0 !== e.pageUrl
                            }
                        }, {
                            key: "keepOpen",
                            get: function () {
                                return !0
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "blacklist"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_blacklist_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_blacklist_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-blacklist-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 20
                            }
                        }]), t
                    }(R),
                    B = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return b.open(this.hit.localFilePath).catch((function (e) {
                                    throw new h.DetailsError(u._("failed_playing_file"), e.message)
                                }))
                            }
                        }, {
                            key: "getReqs",
                            value: function () {
                                return this.reqs.coapp = !0, this.reqs.coappMin = "1.2.4", Promise.resolve()
                            }
                        }, {
                            key: "solveReqs",
                            value: function () {
                                if (this.reqs.coapp) return this.solveCoAppReqs()
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !e.urls && 0 == e.running && !!e.localFilePath
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "openlocalfile"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_openlocalfile_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_openlocalfile_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-play-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 200
                            }
                        }, {
                            key: "catPriority",
                            get: function () {
                                return 1
                            }
                        }]), t
                    }(R),
                    Z = function (e) {
                        function t() {
                            return s(this, t), o(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return a(t, e), i(t, [{
                            key: "doJob",
                            value: function () {
                                return b.open(this.hit.urls && this.hit.localFilePath || this.hit.localDirectory).catch((function (e) {
                                    throw new h.DetailsError(u._("failed_opening_directory"), e.message)
                                }))
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !!e.localDirectory || e.urls && e.localFilePath
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "openlocalcontainer"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return u._("action_openlocalcontainer_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return u._("action_openlocalcontainer_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-open-dir-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 180
                            }
                        }, {
                            key: "catPriority",
                            get: function () {
                                return 1
                            }
                        }]), t
                    }(B),
                    Y = {};

                function J(e) {
                    Object.keys(Y).forEach((function (t) {
                        var n = Y[t];
                        e(n)
                    }))
                }

                function X(e) {
                    Y[e.name] = e
                }
                var G = function (e) {
                    function t(e, n) {
                        s(this, t);
                        var r = o(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                        return r.details = n, d.create(Object.assign(e, {
                            title: n.inFileName
                        })), r
                    }
                    return a(t, e), i(t, [{
                        key: "getReqs",
                        value: function () {
                            var e = this;
                            return e.reqs.convert = e.details.outputConfig, e.reqs.coappMin = "1.2.3", e.downloadWith = "coapp", e.reqs.coapp = !0, e.reqs.checkLicense = !0, Promise.resolve()
                        }
                    }, {
                        key: "getStreams",
                        value: function () {
                            this.streams.full = {
                                type: "full"
                            }
                        }
                    }, {
                        key: "solveStreamName",
                        value: function (e) {
                            var t = this;
                            return Promise.resolve().then((function () {
                                if (t.details.outFileName) return m.call("path.homeJoin", t.details.outDirectory, t.details.outFileName).then((function (e) {
                                    t.filePath = e, t.directory = t.details.outDirectory, t.fileName = t.details.outFileName
                                }));
                                var e = t.details.inFileName,
                                    n = /^(.*)\.([^\.]{1,5})$/.exec(e);
                                return n ? e = n[1] + "." + t.details.extension : e += ".mp4", m.call("makeUniqueFileName", t.details.outDirectory, e).then((function (e) {
                                    t.filePath = e.filePath, t.directory = e.directory, t.fileName = e.fileName
                                }))
                            })).then((function () {
                                return m.call("path.homeJoin", t.details.inDirectory, t.details.inFileName).then((function (t) {
                                    e.filePath = t
                                }))
                            }))
                        }
                    }, {
                        key: "downloadAllStreams",
                        value: function () {
                            return Promise.resolve()
                        }
                    }]), t
                }(M);
                G.idIndex = 0;
                var Q = function (e) {
                    function t(e, n) {
                        s(this, t);
                        var r = o(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                        return r.details = n, d.create(e), r
                    }
                    return a(t, e), i(t, [{
                        key: "getReqs",
                        value: function () {
                            var e = this;
                            e.reqs.coappMin = "1.5.0", e.downloadWith, e.reqs.coapp = !0, e.reqs.checkLicense = !0, e.reqs.aggregate = !0
                        }
                    }, {
                        key: "getStreams",
                        value: function () {
                            this.streams.audio = {
                                type: "audio"
                            }, this.streams.video = {
                                type: "video"
                            }
                        }
                    }, {
                        key: "solveStreamName",
                        value: function (e) {
                            e.fileName = this.details[e.type + "FileName"], e.directory = this.details[e.type + "Directory"], e.filePath = this.details[e.type + "FilePath"]
                        }
                    }, {
                        key: "downloadAllStreams",
                        value: function () {
                            return Promise.resolve()
                        }
                    }, {
                        key: "solveAllReqs",
                        value: function () {
                            var e = this,
                                n = this;
                            return Promise.resolve(r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "solveReqs", this).call(this)).then((function (i) {
                                return i ? r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "solveAllReqs", e).call(e) : n.mergePrepare().then((function () {
                                    return n.reqs.promptFilename = !0, n.reqs.needFilename = !0, Promise.resolve(r(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "solveReqs", e).call(e))
                                }))
                            }))
                        }
                    }, {
                        key: "mergePrepare",
                        value: function () {
                            var e = this,
                                n = void 0,
                                r = void 0,
                                i = void 0,
                                o = void 0,
                                a = void 0,
                                s = void 0,
                                c = "mkv",
                                l = void 0,
                                d = void 0;
                            return g.selectMergeVideoFile(u.prefs.lastDownloadDirectory || "dwhelper").then((function (e) {
                                if (!e) throw new h.VDHError("No video file selected", {
                                    noReport: !0
                                });
                                return n = e.fileName, r = e.directory, i = e.filePath, b.info(i, !0)
                            })).then((function (e) {
                                var t = e.streams.find((function (e) {
                                    return "video" == e.codec_type
                                }));
                                if (!t) throw new h.VDHError(u._("no_video_in_file", [n]));
                                return l = t.codec_name, g.selectMergeAudioFile(r)
                            })).then((function (e) {
                                if (!e) throw new h.VDHError("No audio file selected", {
                                    noReport: !0
                                });
                                return o = e.fileName, a = e.directory, s = e.filePath, b.info(s, !0)
                            })).then((function (f) {
                                var p = {
                                        mpeg4: 1,
                                        h264: 1,
                                        aac: 1,
                                        mp3: 1
                                    },
                                    g = f.streams.find((function (e) {
                                        return "audio" == e.codec_type
                                    }));
                                if (!g) throw new h.VDHError(u._("no_audio_in_file", [o]));
                                d = g.codec_name, p[l] && p[d] && (c = "mp4");
                                ++t.idIndex;
                                var m = n,
                                    v = /^(.*)\.([^\.]{1,5})$/.exec(n);
                                v && (m = v[1]), e.details = {
                                    videoFileName: n,
                                    videoDirectory: r,
                                    videoFilePath: i,
                                    audioFileName: o,
                                    audioDirectory: a,
                                    audioFilePath: s,
                                    outFileName: undefined,
                                    outDirectory: undefined,
                                    outFilePath: undefined,
                                    outExtension: c,
                                    outputConfig: {
                                        format: "mkv"
                                    }
                                }, e.hit.title = m, e.hit.extension = c
                            })).catch((function (e) {
                                throw console.warn("Error merging local", e), e.noReport || g.alert({
                                    title: u._("merge_error"),
                                    text: e.message
                                }), new h.VDHError("Aborted", {
                                    noReport: !0
                                })
                            }))
                        }
                    }]), t
                }(M);

                function $() {
                    u.ui.close("main");
                    var e = "local-merge:" + ++Q.idIndex;
                    return new Q({
                        id: e
                    }).execute()
                }

                function ee() {
                    var e = void 0,
                        t = void 0,
                        n = void 0,
                        r = void 0,
                        i = void 0;
                    return u.ui.close("main"), g.selectConvertFiles(u.prefs.lastDownloadDirectory || "dwhelper").then((function (t) {
                        if (!t) throw new Error("No file selected");
                        return e = t.selected, r = t.directory, n = t.outputConfig, b.getOutputConfigs().then((function (e) {
                            return e[t.outputConfig]
                        }))
                    })).then((function (e) {
                        if (!e) throw new Error("Unknown output config " + n);
                        if (i = (t = e).ext || t.params.f || "mp4", t.audioonly) return v.checkLicense().then((function (e) {
                            if ("accepted" != e.status && "unneeded" != e.status) throw v.alertAudioNeedsReg(), new Error("License required for audio conversion")
                        }))
                    })).then((function () {
                        if (e.length > 1) return g.selectDirectory(r, {
                            titleText: u._("select_output_directory")
                        });
                        var t = e[0],
                            n = /^(.*)\.([^\.]{1,5})$/.exec(t);
                        return n ? t = n[1] + "." + i : t += ".mp4", g.saveAs(t, r)
                    })).then((function (o) {
                        if (!o) return null;
                        u.prefs.dlconvLastOutput = n, e.forEach((function (e) {
                            var n = "local-convert:" + ++G.idIndex;
                            new G({
                                id: n
                            }, {
                                inDirectory: r,
                                inFileName: e,
                                outDirectory: o.directory,
                                outFileName: o.fileName,
                                outputConfig: t,
                                extension: i
                            }).execute()
                        }))
                    })).catch((function (e) {
                        console.warn("Error converting local", e)
                    }))
                }
                Q.idIndex = 0, X(q), X(M), X(F), X(W), X(N), X(U), X(z), X(L), X(V), X(H), X(K), X(B), X(Z), l.listen({
                    convertLocal: ee,
                    mergeLocal: $
                })
            },
            223: (e, t) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.blacklistAdded = function (e) {}, t.downloadSuccess = function (e) {}, t.downloadError = function (e, t) {}
            },
            32: (e, t) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.ReadString = function (e, t) {
                    var n = [];
                    for (; e[t];) n.push(e[t++]);
                    return {
                        string: String.fromCharCode.apply(null, n),
                        length: n.length + 1
                    }
                }, t.ReadInt64 = function (e, n) {
                    var r = t.ReadInt32(e, n),
                        i = t.ReadInt32(e, n + 4);
                    return 4294967296 * r + i
                }, t.ReadInt32 = function (e, t) {
                    return (e[t] << 24) + (e[t + 1] << 16) + (e[t + 2] << 8) + e[t + 3]
                }, t.ReadInt24 = function (e, t) {
                    return (e[t] << 16) + (e[t + 1] << 8) + e[t + 2]
                }, t.ReadInt16 = function (e, t) {
                    return (e[t] << 8) + e[t + 1]
                }, t.ReadInt8 = function (e, t) {
                    return e[t]
                }, t.WriteInt32 = function (e, t, n) {
                    e[t] = (n >>> 24 & 255) >>> 0, e[t + 1] = (n >>> 16 & 255) >>> 0, e[t + 2] = (n >>> 8 & 255) >>> 0, e[t + 3] = (255 & n) >>> 0
                }, t.WriteInt24 = function (e, t, n) {
                    e[t] = (n >>> 16 & 255) >>> 0, e[t + 1] = (n >>> 8 & 255) >>> 0, e[t + 2] = (255 & n) >>> 0
                }, t.WriteInt16 = function (e, t, n) {
                    e[t] = (n >>> 8 & 255) >>> 0, e[t + 1] = (255 & n) >>> 0
                }, t.WriteInt8 = function (e, t, n) {
                    e[t] = (255 & n) >>> 0
                }, t.dump = function (e, t, n) {
                    t = t || 0, n = n || e.length;
                    for (var r = [], i = 0; i < n && i < e.length; i++) {
                        i % 16 == 0 && r.push("\n");
                        var o = e[t + i].toString(16).toUpperCase();
                        if (1 == o.length && (o = "0" + o), r.push(o), (i + 1) % 16 == 0 || i == n - 1 || i == e.length - 1) {
                            for (var a = i + 1; a < (i + 15 & 4294967280); a++) r.push("  ");
                            o = "";
                            for (var s = 4294967280 & i; s <= i; s++) {
                                var u = e[t + s];
                                o += u >= 32 && u < 127 ? String.fromCharCode(u) : "."
                            }
                            r.push(o)
                        }
                    }
                    return r.join(" ")
                }
            },
            224: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.domainsFromHit = h, t.getAllDomains = function () {
                    return Object.keys(u)
                }, t.addToBlacklist = g, t.removeFromBlacklist = m, t.checkHitBlacklisted = v, t.set = y;
                var r = n(3),
                    i = r.browser,
                    o = n(12),
                    a = n(223),
                    s = new RegExp("^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"),
                    u = {};
                i.storage.local.get({
                    blacklist: {}
                }).then((function (e) {
                    u = e.blacklist
                })).catch((function (e) {
                    console.error("Cannot read blacklist storage")
                }));
                var c = null;

                function l() {
                    c = null, i.storage.local.set({
                        blacklist: u
                    }).catch((function (e) {
                        console.error("Cannot write blacklist storage")
                    }))
                }

                function d() {
                    c && clearTimeout(c), c = setTimeout(l, 100)
                }

                function f(e) {
                    var t = [],
                        n = /^https?:\/\/([^\/:]+)/.exec(e);
                    if (n)
                        if (s.test(n[1])) t.push(n[1]);
                        else
                            for (var r = n[1].split("."); r.length > 1 && ("co" != r[0] || r.length > 2);) t.push(r.join(".")), r.shift();
                    return t
                }

                function p(e) {
                    var t = [];
                    e.url && (t = t.concat(f(e.url))), e.audioUrl && (t = t.concat(f(e.audioUrl))), e.videoUrl && (t = t.concat(f(e.videoUrl))), e.topUrl && (t = t.concat(f(e.topUrl))), e.pageUrl && (t = t.concat(f(e.pageUrl)));
                    var n = {};
                    return t.forEach((function (e) {
                        n[e] = 1
                    })), n
                }

                function h(e) {
                    return function (e) {
                        var t = Object.keys(e);
                        return t.sort((function (e, t) {
                            for (var n = e.split(".").reverse(), r = t.split(".").reverse();;) {
                                if (n.length && !r.length) return -1;
                                if (!n.length && r.length) return 1;
                                if (!n.length && !r.length) return 0;
                                var i = n.shift(),
                                    o = r.shift();
                                if (i != o) return i < o ? -1 : 1
                            }
                        })), t
                    }(p(e))
                }

                function g(e) {
                    e.forEach((function (e) {
                        u[e] = !0, a.blacklistAdded(e)
                    }));
                    var t = [],
                        n = o.getHits();
                    Object.keys(n).forEach((function (e) {
                        v(n[e]) && t.push(e)
                    })), t.length > 0 && o.dispatch("hit.delete", t), d()
                }

                function m(e) {
                    e.forEach((function (e) {
                        delete u[e]
                    })), d()
                }

                function v(e) {
                    if (!r.prefs.blacklistEnabled) return !1;
                    var t = p(e);
                    for (var n in t)
                        if (u[n]) return !0;
                    return !1
                }

                function y(e) {
                    u = e || {}, d()
                }
                r.rpc.listen({
                    domainsFromHitId: function (e) {
                        var t = o.getHit(e);
                        return t && h(t) || []
                    },
                    addToBlacklist: g,
                    removeFromBlacklist: m,
                    setBlacklist: function (e) {
                        return y(e)
                    },
                    getBlacklist: function () {
                        return Object.keys(u).filter((function (e) {
                            return !!u[e]
                        }))
                    },
                    editBlacklist: function () {
                        r.ui.open("blacklist-edit", {
                            type: "tab",
                            url: "content/blacklist-edit.html"
                        })
                    }
                })
            },
            217: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
                        return typeof e
                    } : function (e) {
                        return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                    },
                    i = function () {
                        function e(e, t) {
                            for (var n = 0; n < t.length; n++) {
                                var r = t[n];
                                r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                            }
                        }
                        return function (t, n, r) {
                            return n && e(t.prototype, n), r && e(t, r), t
                        }
                    }();

                function o(e, t) {
                    if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                    return !t || "object" != typeof t && "function" != typeof t ? e : t
                }

                function a(e, t) {
                    if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                    e.prototype = Object.create(t && t.prototype, {
                        constructor: {
                            value: e,
                            enumerable: !1,
                            writable: !0,
                            configurable: !0
                        }
                    }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                }

                function s(e, t) {
                    if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                }
                t.networkHook = function (e, t) {
                    if (!u.prefs.chunksEnabled) return null;
                    var n = null;
                    u.prefs.dashEnabled && (v.test(e.url) ? n = new x(e.url, "json", t) : t.contentType && y.test(t.contentType.toLowerCase()) && (n = new x(e.url, "xml", t)));
                    u.prefs.hlsEnabled && (b.test(e.url) ? n = new A(e.url) : w.test(e.url) ? n = new A(e.url, "json") : t.contentType && t.contentType.toLowerCase().indexOf("mpegurl") >= 0 && (n = new A(e.url)));
                    if (n && !n.skipManifest) return new Promise((function (r, i) {
                        if ("GET" != e.method) return i(new Error("Not a GET request getting chunks manifest"));
                        u.prefs.chunkedCoappManifestsRequests || g.isProbablyAvailable() ? g.request(e.url, {
                            headers: t.headers,
                            proxy: u.prefs.coappUseProxy && t.proxy || null
                        }).then((function (e) {
                            n.handleManifest(e), n.checkReady(), r(n)
                        })).catch(i) : h.request({
                            url: e.url
                        }).then((function (t) {
                            t.ok ? t.text().then((function (e) {
                                n.handleManifest(e), n.checkReady(), r(n)
                            })) : (console.warn("Error retrieving manifest from", e.url), r(null))
                        })).catch((function (t) {
                            console.warn("Error retrieving manifest from", e.url), r(null)
                        }))
                    }));
                    return null
                }, t.download = function (e, t, n, i, o) {
                    t.ignoreSpecs = !0;
                    var a = null;
                    switch (e.hit.chunked) {
                        case "hls":
                            a = l.getChunkSet(e.hit);
                            break;
                        case "dash-adp":
                            if (t.length > 1) return void
                            function (e, t, n, i, o) {
                                var a = e.hit,
                                    s = {
                                        audio: 0,
                                        video: 0
                                    },
                                    c = {
                                        audio: !1,
                                        video: !1
                                    },
                                    l = !1;

                                function d(e) {
                                    l || (l = !0, i(e))
                                }
                                var p = {},
                                    m = !1;
                                t.forEach((function (t) {
                                    var i = t.type;

                                    function l() {
                                        c[i] = !0, c.audio && c.video && n()
                                    }

                                    function v(e) {
                                        s[i] = e, m || o(Math.round((s.audio + s.video) / 2))
                                    }
                                    try {
                                        var y = function (n) {
                                                var r = a.url || a[i + "Url"];
                                                a._mpdCommonBaseUrl && (r = new URL(a._mpdCommonBaseUrl, r).href), w.base_url && (r = new URL(w.base_url, r).href), b = new f.DashChunkset({
                                                    url: r,
                                                    headers: e.hit.headers || [],
                                                    proxy: u.prefs.coappUseProxy && e.hit.proxy || null
                                                }), p[i] = {
                                                    chunkset: b,
                                                    target: t.fileName
                                                }, b.downloadFile(t.fileName, {
                                                    init_segment: n,
                                                    segments: w.segments
                                                }, r, l, d, v)
                                            },
                                            b = void 0,
                                            w = a[i + "Mpd"];
                                        if ("string" == typeof w.init_segment) y(h.toByteArray(w.init_segment));
                                        else if ("object" == r(w.init_segment)) y(w.init_segment);
                                        else {
                                            var k = new URL(w.init_segment_url, a[i + "Url"]).href;
                                            u.prefs.chunkedCoappDataRequests ? g.requestBinary(k, {
                                                headers: a.headers || [],
                                                proxy: u.prefs.coappUseProxy && a.proxy || null
                                            }).then((function (e) {
                                                y(e)
                                            })).catch((function (e) {
                                                d(e.message)
                                            })) : h.downloadToByteArray(k, a.headers, a.isPrivate).then((function (e) {
                                                y(e)
                                            })).catch((function (e) {
                                                d(e.message)
                                            }))
                                        }
                                    } catch (e) {
                                        console.warn("Error", e), d(new Error("Dash ADP: " + e.message))
                                    }
                                })), e.abortChunked = function () {
                                    for (var e in m = !0, p) p[e].chunkset.actionAbortFn();
                                    i(new h.VDHError("User abort", {
                                        noReport: !0
                                    }))
                                }
                            }(e, t, n, i, o);
                        case "dash":
                            a = new f.DashChunkset(e.hit)
                    }
                    if (!a) return void m.error("Requested download of chunked stream, but no chunkset found");
                    a.download(e, t, n, i, o)
                };
                var u = n(3),
                    c = n(12),
                    l = n(218),
                    d = n(221),
                    f = n(241),
                    p = n(196),
                    h = n(7),
                    g = n(34),
                    m = n(190),
                    v = new RegExp("^https?://.*/master\\.json"),
                    y = new RegExp("dash.*mpd"),
                    b = new RegExp("^https?://.*\\.m3u8(?:\\?|$)"),
                    w = new RegExp("^https?://api\\.periscope\\.tv/api/v2/getAccessPublic");
                var k = function () {
                        function e(t) {
                            s(this, e), this.type = t, this.receivedChunks = []
                        }
                        return i(e, [{
                            key: "handleHit",
                            value: function (e) {
                                this.hitData = e, this.checkReady()
                            }
                        }, {
                            key: "checkReady",
                            value: function () {}
                        }, {
                            key: "handleManifest",
                            value: function () {}
                        }, {
                            key: "handle",
                            value: function () {}
                        }]), e
                    }(),
                    x = function (e) {
                        function t(e, n, r) {
                            s(this, t);
                            var i = o(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, "dash"));
                            return i.format = n, i.manifestUrl = e, i.meta = r, i
                        }
                        return a(t, e), i(t, [{
                            key: "handleManifest",
                            value: function (e) {
                                try {
                                    if ("json" == this.format) {
                                        var t = JSON.parse(e);
                                        t && Array.isArray(t.video) && t.video.length > 0 && Array.isArray(t.video[0].segments) && t.video[0].segments.length > 0 && (this.mpd = t)
                                    } else "xml" == this.format && (this.handler = p.dashManifest(this.manifestUrl, e, this.meta))
                                } catch (e) {
                                    console.error("Error parsing DASH manifest", e.message || e)
                                }
                            }
                        }, {
                            key: "checkReady",
                            value: function () {
                                this.hitData && (this.mpd || this.handler) && this.handle()
                            }
                        }, {
                            key: "pickAudioMpd",
                            value: function () {
                                var e = [{
                                        field: "codecs",
                                        pref: "mp4a.40.2"
                                    }, {
                                        field: "format",
                                        pref: "mp42"
                                    }, {
                                        field: "mime_type",
                                        pref: "audio/mp4"
                                    }, {
                                        field: "channels",
                                        pref: "@max"
                                    }, {
                                        field: "bitrate",
                                        pref: "@max"
                                    }, {
                                        field: "sample_rate",
                                        pref: "@max"
                                    }],
                                    t = [].concat(this.mpd.audio);
                                return t.sort((function (t, n) {
                                    for (var r = 0; r < e.length; r++) {
                                        var i = e[r];
                                        if (t[i.field] != n[i.field]) {
                                            if ("@max" == i.pref) return n[i.field] - t[i.field];
                                            if (t[i.field] == i.pref) return -1;
                                            if (n[i.field] == i.pref) return 1
                                        }
                                    }
                                    return 0
                                })), t[0]
                            }
                        }, {
                            key: "handle",
                            value: function () {
                                var e = h.hashHex(this.hitData.url),
                                    t = this;
                                if (this.handler) this.handler(this.hitData);
                                else if (this.mpd) {
                                    var n = t.mpd.audio && t.mpd.audio.length > 0 && Array.isArray(t.mpd.audio[0].segments) && t.mpd.audio[0].segments.length > 0,
                                        r = t.mpd.video && t.mpd.video.length > 0 && Array.isArray(t.mpd.video[0].segments) && t.mpd.video[0].segments.length > 0,
                                        i = this.mpd.video,
                                        o = this.mpd.audio;
                                    !r || n && "audio" == u.prefs.dashOnAdp ? (i = this.mpd.audio, o = null) : (!n || r && "video" == u.prefs.dashOnAdp) && (o = null), i.forEach((function (n, r) {
                                        var i = {};
                                        o ? (i.chunked = "dash-adp", i.audioMpd = t.pickAudioMpd(), i.videoMpd = n, i.audioUrl = new URL("dash-audio.mp4", t.hitData.url).href, i.videoUrl = new URL("dash-video.mp4", t.hitData.url).href, i.url = void 0) : i._mpd = n;
                                        var a = Object.assign({}, t.hitData, {
                                            id: "dash:" + e + "-" + r,
                                            extension: "mp4",
                                            bitrate: n.bitrate || n.avg_bitrate || null,
                                            length: null,
                                            chunked: "dash",
                                            descrPrefix: u._("dash_streaming"),
                                            group: "grp-" + e
                                        }, i);
                                        a._mpdCommonBaseUrl = t.mpd.base_url, n.width && n.height && (a.size = n.width + "x" + n.height), n.duration && (a.duration = Math.round(n.duration)), c.dispatch("hit.new", a)
                                    }))
                                }
                            }
                        }]), t
                    }(k),
                    A = function (e) {
                        function t(e, n) {
                            s(this, t);
                            var r = o(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, "hls"));
                            return r.masterFormat = n || "m3u8", r.mediaUrl = e, r
                        }
                        return a(t, e), i(t, [{
                            key: "handleHit",
                            value: function (e) {
                                e.group = "grp-" + h.hashHex(e.url), e.masterManifest = e.url, k.prototype.handleHit.call(this, e)
                            }
                        }, {
                            key: "handleManifest",
                            value: function (e) {
                                var t = null;
                                "m3u8" == this.masterFormat ? t = d.get(e, this.mediaUrl) : "json" == this.masterFormat && (t = d.getPsJson(e, this.mediaUrl)), t && (t.isMaster() ? this.master = t : t.isMedia() && (this.media = t))
                            }
                        }, {
                            key: "checkReady",
                            value: function () {
                                this.hitData && (this.master || this.media) && this.handle()
                            }
                        }, {
                            key: "handle",
                            value: function () {
                                this.master ? l.handleMaster(this.master, this.hitData) : this.media && l.handleMedia(this.media, this.hitData, this.mediaUrl)
                            }
                        }]), t
                    }(k)
            },
            220: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = function () {
                    function e(e, t) {
                        for (var n = 0; n < t.length; n++) {
                            var r = t[n];
                            r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                        }
                    }
                    return function (t, n, r) {
                        return n && e(t.prototype, n), r && e(t, r), t
                    }
                }();
                var i = n(3),
                    o = n(78),
                    a = n(50),
                    s = n(190),
                    u = n(7),
                    c = n(219),
                    l = n(12),
                    d = n(34);
                t.Codecs = {
                    27: {
                        id: 27,
                        type: "video",
                        name: "h264",
                        strTag: "avc1",
                        tag: 1635148593,
                        captureRaw: !0
                    },
                    15: {
                        id: 15,
                        type: "audio",
                        name: "aac",
                        strTag: "mp4a",
                        tag: 1836069985,
                        captureRaw: !0
                    }
                }, t.Chunkset = function () {
                    function e(t) {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, e);
                        var n = t && t.url && u.hashHex(t.url) || "nohash";
                        this.id = "chunked:" + n, this.chunks = [], this.hash = n, this.recording = !1, this.lastDlingIndex = -1, this.lastDledIndex = -1, this.lastProcedIndex = -1, this.downloadingCount = 0, this.nextTrackId = 1, this.lastProgress = -1, this.processedChunksCount = 0, this.dataWritten = 0, this.currentDataBlockSize = -1, this.fileSize = 0, this.mdatOffsets = [], this.multiMdat = !1, this.rawAppendData = !1, this.hit = Object.assign({}, t, {
                            id: this.id,
                            length: 0,
                            url: t && t.url || null
                        })
                    }
                    return r(e, [{
                        key: "updateHit",
                        value: function (e) {
                            this.hit.length = (this.hit.length || 0) + (e.length || 0), a.newData(this.hit), this.chunks.push({
                                url: e.url,
                                index: this.chunks.length
                            })
                        }
                    }, {
                        key: "handle",
                        value: function () {
                            if (this.recording) {
                                for (; this.downloadingCount < i.prefs.chunksConcurrentDownloads && this.lastDlingIndex + 1 < this.chunks.length && this.lastDledIndex - this.lastProcedIndex < i.prefs.chunksPrefetchCount;) this.progressFn && this.lastProgress < 0 && (this.lastProgress = 0, this.progressFn(0)), this.downloadingCount++, this.downloadChunk.call(this, this.chunks[++this.lastDlingIndex], (function (e, t) {
                                    if (this.downloadingCount--, e) {
                                        if (!this.recording) return;
                                        this.stopDownloadingOnChunkError ? (this.noMoreChunkToDownload = !0, t.index < this.chunks.length && (this.chunks.splice(t.index), t.index > 0 ? null === this.chunks[t.index - 1] && (this.recording = !1, this.finalize(null)) : this.recording && (this.recording = !1, this.finalize(new Error("No chunk received")))), this.handle()) : (this.recording = !1, this.lastDlingIndex = this.lastDledIndex, this.doNotReportDownloadChunkErrors && this.lastDledIndex >= 0 ? this.finalize(null, (function () {})) : (console.warn("Error downloading chunk:", e.message || e), s.error(e), this.aborted = !0, this.finalize(e, (function () {}))))
                                    } else
                                        for (t.downloading = !1, t.downloaded = !0; this.lastDledIndex + 1 < this.chunks.length && this.chunks[this.lastDledIndex + 1].downloaded;) this.lastDledIndex++;
                                    this.aborted ? t.path && (c.File.remove(t.path), delete t.path) : this.handle()
                                }));
                                this.recording && this.lastProcedIndex < this.lastDledIndex && this.lastProcedIndex < this.chunks.length - 1 && !this.chunks[this.lastProcedIndex + 1].processing && this.processChunk.call(this, this.chunks[this.lastProcedIndex + 1], (function (e, t) {
                                    e && console.warn("Error processing chunk: move to next chunk", e), t.processing = !1, this.lastProcedIndex = t.index, t.path && c.File.remove(t.path), this.handle(), this.chunks[t.index] = null
                                }))
                            }
                        }
                    }, {
                        key: "downloadChunk",
                        value: function (e, t) {
                            if (e.downloaded) return t.call(this, null, e);
                            e.downloading = !0;
                            var n = this;

                            function r(t) {
                                i.prefs.chunkedCoappDataRequests ? d.requestBinary(e.url, {
                                    headers: n.hit.headers || [],
                                    proxy: i.prefs.coappUseProxy && n.hit.proxy || null
                                }).then((function (r) {
                                    e.data = r, t.call(n, null, e)
                                })).catch((function (r) {
                                    t.call(n, r, e)
                                })) : u.downloadToByteArray(e.url, n.hit.headers || null, n.hit.isPrivate).then((function (r) {
                                    e.data = r, t.call(n, null, e)
                                })).catch((function (r) {
                                    t.call(n, r, e)
                                }))
                            }
                            e.downloadRetries = 0, r((function e(o, a) {
                                o && a.downloadRetries++ <= i.prefs.downloadRetries ? setTimeout((function () {
                                    n.recording && r(e)
                                }), i.prefs.downloadRetryDelay) : (delete a.downloadRetries, t.call(n, o, a))
                            }))
                        }
                    }, {
                        key: "processChunkData",
                        value: function (e, t) {
                            t.call(this, null, e)
                        }
                    }, {
                        key: "processChunk",
                        value: function (e, t) {
                            var n = this;

                            function r(r) {
                                function i() {
                                    n.processChunkData(r, (function (r, i) {
                                        r ? t.call(n, r, e) : n.appendDataToOutputFile(i, (function (r) {
                                            if (r || (n.dataWritten += o.length(i)), n.processedSegmentsCount++, n.processedSegmentsCount >= n.segmentsCount) n.outOfChunks();
                                            else if (n.progressFn && !n.aborted) {
                                                var a = Math.round(100 * n.processedSegmentsCount / (n.segmentsCount || n.chunks.length || 1));
                                                a != n.lastProgress && n.progressFn(a), n.lastProgress = a
                                            }
                                            t.call(n, r, e)
                                        }))
                                    }))
                                }
                                n.endsOnSeenChunk ? crypto.subtle.digest({
                                    name: "SHA-256"
                                }, r).then((function (e) {
                                    for (var t = [], r = new DataView(e), o = 0; o < r.byteLength; o += 4) {
                                        var a = "00000000",
                                            s = (a + r.getUint32(o).toString(16)).slice(-8);
                                        t.push(s)
                                    }
                                    var u = t.join("");
                                    if (n.seenChunks = n.seenChunks || {}, n.seenChunks[u]) return n.recording = !1, void n.finalize(null);
                                    n.seenChunks[u] = !0, i()
                                })) : i()
                            }
                            e.processing = !0, e.data ? r(e.data) : c.File.read(e.path).then((function (e) {
                                r(e)
                            }), (function (r) {
                                t.call(n, r, e)
                            }))
                        }
                    }, {
                        key: "outOfChunks",
                        value: function () {
                            this.recording = !1, this.finalize(null)
                        }
                    }, {
                        key: "download",
                        value: function (e, t, n, r, i) {
                            var a = this;
                            this.downloadTarget = t[0].fileName, this.aborted = !1, this.action = e, o.writeFileHeader(this, (function (e) {
                                e ? r(e) : (a.recording = !0, a.handle())
                            }))
                        }
                    }, {
                        key: "getNextTrackId",
                        value: function () {
                            return this.nextTrackId
                        }
                    }, {
                        key: "setNewId",
                        value: function () {
                            for (var e = 1; l.getHit(this.id + "-" + e);) e++;
                            this.id = this.id + "-" + e, this.hit && (this.hit.id = this.id), this.action && this.action.hit && (this.action.hit.id = this.id)
                        }
                    }, {
                        key: "finalize",
                        value: function (e, t) {
                            if (this.cleanupChunkFiles(), this.progressFn && this.progressFn(100), e && this.errorFn ? this.errorFn(e) : !e && this.successFn && this.successFn(), !e) {
                                var n = l.getHit(this.id);
                                if (n) {
                                    var r = Object.assign({}, n);
                                    this.hit = r, delete r.url, l.dispatch("hit.delete", this.id), this.setNewId(), r.id = this.id, l.dispatch("hit.new", r)
                                }
                            }
                            t && t(e)
                        }
                    }, {
                        key: "appendDataToOutputFile",
                        value: function (e, t) {
                            var n = this,
                                r = o.length(e);

                            function i() {
                                n.currentDataBlockSize += r, n.appendToOutputFile(e, (function (e) {
                                    if (e) return t(e);
                                    t(null)
                                }))
                            }

                            function a() {
                                n.mdatOffsets.push(n.lastDataIndex + 8);
                                var e = o.mdatBox();
                                n.appendToOutputFile(e, (function (e, o) {
                                    if (e) return t(e);
                                    n.lastDataIndex = o + r, n.mdatLengthOffset = o - 8, n.currentDataBlockSize = 0, i()
                                }))
                            }
                            this.rawAppendData ? (this.mdatOffsets.push(this.lastDataIndex), i(), this.lastDataIndex += r) : this.currentDataBlockSize < 0 ? a() : this.currentDataBlockSize + r > 1e9 ? (this.multiMdat = !0, o.updateMdatLength(this, this.mdatLengthOffset, this.currentDataBlockSize, (function (e) {
                                if (e) return t(e);
                                a()
                            }))) : (this.mdatOffsets.push(this.lastDataIndex), i(), this.lastDataIndex += r)
                        }
                    }, {
                        key: "appendToOutputFile",
                        value: function (e, t) {
                            var n = this;
                            if (this.aborted) return t(null);

                            function r() {
                                if (n.aborted)
                                    for (; n.pendingAppend.length > 0;) {
                                        n.pendingAppend.shift().callback(null)
                                    } else {
                                        n.appendFileTimer && clearTimeout(n.appendFileTimer), n.appendFileTimer = setTimeout((function () {
                                            n.file.close(), n.file = null
                                        }), 5e3);
                                        for (var t = 0; n.pendingAppend.length > 0;) {
                                            var r = n.pendingAppend.shift();
                                            t++,
                                            function (r) {
                                                o.writeMulti(n.file, r.data, (function (i) {
                                                    var a = o.length(e);
                                                    if (n.fileSize += a, r.callback(i, n.fileSize), 0 == --t && n.waitingDataWritten)
                                                        for (; n.waitingDataWritten.length;) n.waitingDataWritten.shift()()
                                                }))
                                            }(r)
                                        }
                                    }
                            }
                            if (this.pendingAppend = this.pendingAppend || [], this.pendingAppend.push({
                                    data: e,
                                    callback: t
                                }), this.file) r();
                            else if (!this.openingAppendFile) {
                                this.openingAppendFile = !0;
                                var i = n.downloadTarget;
                                this.rawAppendData || (i += ".part"), c.File.open(i, {
                                    write: !0,
                                    append: !0
                                }).then((function (e) {
                                    n.openingAppendFile = !1, n.file = e, r()
                                }), (function (e) {
                                    for (n.openingAppendFile = !1; n.pendingAppend.length > 0;) {
                                        n.pendingAppend.shift().callback(e)
                                    }
                                }))
                            }
                        }
                    }, {
                        key: "waitForWrittenData",
                        value: function (e) {
                            this.aborted ? e() : this.pendingAppend && this.pendingAppend.length ? (this.waitingDataWritten = this.waitingDataWritten || [], this.waitingDataWritten.push(e)) : e()
                        }
                    }, {
                        key: "cleanupChunkFiles",
                        value: function () {
                            for (var e = Math.max(0, this.lastProcedIndex); e < Math.max(0, this.lastDledIndex); e++) {
                                var t = this.chunks[e];
                                t && t.path && (c.File.remove(t.path), t = null)
                            }
                        }
                    }, {
                        key: "actionAbortFn",
                        value: function () {
                            this.recording = !1, this.aborted = !0
                        }
                    }]), e
                }()
            },
            237: (e, t) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var n = function () {
                    function e(e, t) {
                        for (var n = 0; n < t.length; n++) {
                            var r = t[n];
                            r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                        }
                    }
                    return function (t, n, r) {
                        return n && e(t.prototype, n), r && e(t, r), t
                    }
                }();
                t.Downloads = function () {
                    function e(t) {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, e), this.coapp = t
                    }
                    return n(e, [{
                        key: "download",
                        value: function (e) {
                            return this.coapp.call("downloads.download", e)
                        }
                    }, {
                        key: "search",
                        value: function (e) {
                            return this.coapp.call("downloads.search", e)
                        }
                    }, {
                        key: "cancel",
                        value: function (e) {
                            return this.coapp.call("downloads.cancel", e)
                        }
                    }]), e
                }()
            },
            34: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.gotoInstall = p, t.call = h, t.listen = function () {
                    return a.listen.apply(a, arguments)
                }, t.check = g, t.isProbablyAvailable = function () {
                    return f
                }, t.request = function (e, t) {
                    return new Promise((function (n, r) {
                        var i = [];
                        a.call("request", e, t).then((function (e) {
                            return f = !0, e
                        })).then((function e(t) {
                            if (i.push(t.data), !t.more) return n(i.join(""));
                            a.call("requestExtra", t.id).then((function (t) {
                                e(t)
                            })).catch(r)
                        })).catch(r)
                    }))
                }, t.requestBinary = function (e, t) {
                    return new Promise((function (n, r) {
                        var i = 0,
                            o = [];
                        a.call("requestBinary", e, t).then((function (e) {
                            return f = !0, e
                        })).then((function e(t) {
                            if (t.data && t.data.data && (i += t.data.data.length, o.push(new Uint8Array(t.data.data))), !t.more) {
                                var s = new Uint8Array(i),
                                    u = 0;
                                return o.forEach((function (e) {
                                    s.set(e, u), u += e.length
                                })), n(s)
                            }
                            a.call("requestExtra", t.id).then((function (t) {
                                setTimeout((function () {
                                    e(t)
                                }))
                            })).catch(r)
                        })).catch(r)
                    }))
                };
                var r = n(3),
                    i = n(7),
                    o = n(237),
                    a = n(215)("net.downloadhelper.coapp"),
                    s = n(49),
                    u = i.Concurrent(),
                    c = i.Concurrent(),
                    l = n(8).buildOptions || {},
                    d = null,
                    f = void 0;

                function p() {
                    u((function () {
                        var e = "https://www.downloadhelper.net/install-coapp";
                        return e += "?browser=" + (l.browser || ""), r.prefs.forcedCoappVersion && (e += "version=" + r.prefs.forcedCoappVersion), s.gotoOrOpenTab(e)
                    }))
                }

                function h() {
                    return a.call.apply(a, arguments)
                }

                function g() {
                    return c((function () {
                        return new Promise((function (e, t) {
                            var n = !1;
                            a.callCatchAppNotFound((function (t) {
                                f = !1, n = !0, e({
                                    status: !1,
                                    error: t.message
                                })
                            }), "info").then((function (t) {
                                f = !0, e({
                                    status: !0,
                                    info: t
                                })
                            })).catch((function (t) {
                                f = !1, n || e({
                                    status: !1,
                                    error: t.message
                                })
                            }))
                        }))
                    }))
                }
                a.onAppNotFound.addListener((function () {
                    p()
                })), a.onCallCount.addListener((function (e, t) {
                    d && (clearTimeout(d), d = null), 0 === e && 0 === t && r.prefs.coappIdleExit && (d = setTimeout((function () {
                        d = null, a.close()
                    }), r.prefs.coappIdleExit))
                }));
                t.downloads = new o.Downloads(a);
                r.prefs.checkCoappOnStartup && setTimeout((function () {
                    g()
                }), 1e3), r.rpc.listen({
                    coappProxy: h,
                    checkCoApp: g,
                    installCoApp: p
                })
            },
            192: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.updateHit = function (e) {
                    console.warn("TODO converter.updateHit")
                }, t.info = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] && arguments[1],
                        n = o.call("probe", e, t);
                    return t ? n.then((function (e) {
                        return JSON.parse(e)
                    })) : n
                }, t.play = function (e) {
                    return o.call("play", e)
                }, t.open = function (e) {
                    return o.call("open", e)
                }, t.wmForHeight = function (e) {
                    return new Promise((function (t, n) {
                        var r = 1 / 0,
                            i = null;
                        for (var a in m) {
                            var s = parseInt(a),
                                u = Math.abs(e - s);
                            u < r && (i = a, r = u)
                        }
                        var c = void 0,
                            l = void 0,
                            d = m[i];
                        o.call("tmp.file", {
                            prefix: "vdh-wm-",
                            postfix: ".gif"
                        }).then((function (e) {
                            var t = e.path,
                                n = e.fd;
                            c = t, l = n;
                            for (var r = atob(d.qr), i = new Array(r.length), a = 0; a < r.length; a++) i[a] = r.charCodeAt(a);
                            return o.call("fs.write", l, i)
                        })).then((function () {
                            return o.call("fs.close", l)
                        })).then((function () {
                            t({
                                x: d.x,
                                y: d.y,
                                qr: c
                            })
                        })).catch(n)
                    }))
                }, t.makeUniqueFileName = function (e) {
                    return o.call("makeUniqueFileName", e)
                }, t.convert = function (e, t) {
                    return new Promise((function (n, i) {
                        var c = ["-y", "-i", e.source];
                        for (var l in e.wm && e.wm.qr && !e.config.audioonly && (c = c.concat(["-i", e.wm.qr, "-filter_complex", "[0:v][1:v] overlay=" + e.wm.x + ":" + e.wm.y])), e.config.params) {
                            var d = e.config.params[l];
                            null !== d && ("string" != typeof d || d.length > 0) && (c.push("-" + l), c.push("" + d))
                        }
                        e.config.extra && /^\s*(.*?)\s*$/.exec(e.config.extra)[1].split(/\s+/).forEach((function (e) {
                            c.push(e)
                        }));
                        e.config.audioonly && c.push("-vn"), c = c.concat(["-threads", r.prefs.converterThreads, "-strict", "experimental"]);
                        var f = ++s;
                        u[f] = t, e.config.twopasses ? console.warn("TODO implement 2 passes conversion") : (c.push(e.target), o.call("convert", c, {
                            progressTime: "" + f
                        }).then((function (t) {
                            var o = t.exitCode,
                                s = t.stderr;
                            delete u[f], 0 !== o ? i(new a.DetailsError(r._("failed_converting", e.source), s)) : n()
                        })).catch((function (e) {
                            delete u[f], i(e)
                        })))
                    }))
                }, t.aggregate = function (e, t) {
                    return new Promise((function (n, i) {
                        var c = ["-y", "-i", e.audio, "-i", e.video],
                            l = !1;
                        "h264" == e.videoCodec && r.prefs.converterAggregTuneH264 && (l = !0, e.videoCodec = "libx264"), e.wm && e.videoCodec && e.fps ? c = c.concat(["-i", e.wm.qr, "-filter_complex", "[1:v][2:v] overlay=" + e.wm.x + ":" + e.wm.y, "-c:v", e.videoCodec]) : (c.push("-c:v"), l ? c.push(e.videoCodec) : c.push("copy")), c = c.concat(["-map", "0:a", "-map", "1:v"]), l && (c = c.concat(["-preset", "fast", "-tune", "film", "-profile:v", "baseline", "-level", "30"])), c = (c = c.concat(["-g", "9999"])).concat(["-c:a", "copy", "-threads", r.prefs.converterThreads, "-strict", "experimental", e.target]);
                        var d = ++s;
                        u[d] = t, o.call("convert", c, {
                            progressTime: "" + d
                        }).then((function (e) {
                            if (delete u[d], 0 !== e.exitCode) throw new a.DetailsError("Failed conversion", e.stderr);
                            n()
                        })).catch((function (e) {
                            delete u[d], i(e)
                        }))
                    }))
                }, t.setOutputConfigs = d, t.resetOutputConfigs = f, t.getFormats = p, t.getCodecs = h;
                var r = n(3),
                    i = r.browser,
                    o = n(34),
                    a = n(7);
                var s = 0,
                    u = {};
                var c = new a.Cache((function () {
                        return i.storage.local.get("outputConfigs").then((function (e) {
                            return e.outputConfigs || g
                        }))
                    }), (function (e) {
                        return i.storage.local.set({
                            outputConfigs: e
                        })
                    })),
                    l = t.getOutputConfigs = c.get();

                function d(e) {
                    return c.set(Object.assign({}, g, e))
                }

                function f() {
                    return l().then((function (e) {
                        var t = Object.assign({}, e);
                        return Object.keys(t).forEach((function (e) {
                            t[e].readonly || delete t[e]
                        })), c.set(t)
                    }))
                }

                function p() {
                    return o.call("formats")
                }

                function h() {
                    return o.call("codecs")
                }
                r.rpc.listen({
                    getOutputConfigs: l,
                    setOutputConfigs: d,
                    resetOutputConfigs: f,
                    editConverterConfigs: function (e) {
                        r.ui.open("convoutput" + (e ? "#" + e : ""), {
                            type: "tab",
                            url: "content/convoutput.html"
                        })
                    },
                    getFormats: p,
                    getCodecs: h
                }), o.listen({
                    convertOutput: function (e, t) {
                        var n = u[e];
                        n && n(t)
                    }
                });
                var g = {
                        "e6587753-4ca5-4d2e-b7ba-beaf1e7f191c": {
                            title: "Re-encoded MP4 (h264/aac)",
                            ext: "mp4",
                            params: {
                                "c:a": "aac",
                                f: "mp4",
                                "c:v": "h264"
                            },
                            audioonly: !1,
                            readonly: !0
                        },
                        "249a7d34-3640-4ac3-8300-13827811d2cf": {
                            title: "MPEG (mpeg1+mp2)",
                            ext: "mpg",
                            params: {
                                "c:a": "mp2",
                                f: "mpeg",
                                r: 24,
                                "c:v": "mpeg1video"
                            },
                            extra: "-mbd rd -trellis 2 -cmp 2 -subcmp 2 -g 100",
                            audioonly: !1,
                            readonly: !0
                        },
                        "6de4f5ce-8cfe-4f0f-8246-bacb7b0d7624": {
                            title: "WMV 500Kb (Windows Media Player)",
                            ext: "wmv",
                            params: {
                                "c:a": "wmav2",
                                f: "asf",
                                "c:v": "wmv2",
                                "b:v": "500k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "21a19146-e116-4460-8356-a8eab9cf61ce": {
                            title: "WMV 1Mb (Windows Media Player)",
                            ext: "wmv",
                            params: {
                                "c:a": "wmav2",
                                f: "asf",
                                "c:v": "wmv2",
                                "b:v": "1000k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "933b1b41-6862-4ce0-9605-10fa5e4b310c": {
                            title: "WMV 2Mb (Windows Media Player)",
                            ext: "wmv",
                            params: {
                                "c:a": "wmav2",
                                f: "asf",
                                "c:v": "wmv2",
                                "b:v": "2000k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "90195ab2-d891-443c-a164-8f0953ec8975": {
                            title: "WMV 4Mb (Windows Media Player)",
                            ext: "wmv",
                            params: {
                                "c:a": "wmav2",
                                f: "asf",
                                "c:v": "wmv2",
                                "b:v": "4000k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "3a4cc0a6-6eb0-4cff-90fb-fdf8eb6a9571": {
                            title: "AVI 500Kb (mpeg4/mp3)",
                            ext: "avi",
                            params: {
                                "c:a": "mp3",
                                f: "avi",
                                "c:v": "mpeg4",
                                "b:v": "500k",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "ebdbb895-7a1e-43e2-bef4-be6e62cb8507": {
                            title: "AVI 1Mb (mpeg4/mp3)",
                            ext: "avi",
                            params: {
                                "c:a": "mp3",
                                f: "avi",
                                "c:v": "mpeg4",
                                "b:v": "1000k",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "0b6280d3-f8f2-4cb6-8235-a5a4b91488f7": {
                            title: "AVI 2Mb (mpeg4/mp3)",
                            ext: "avi",
                            params: {
                                "c:a": "mp3",
                                f: "avi",
                                "c:v": "mpeg4",
                                "b:v": "2000k",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "9ea8a22b-5738-4d0f-8494-3037ec568191": {
                            title: "AVI 4Mb (mpeg4/mp3)",
                            ext: "avi",
                            params: {
                                "c:a": "mp3",
                                f: "avi",
                                "c:v": "mpeg4",
                                "b:v": "4000k",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "4174b9dd-c2a0-409d-801d-c84f96be0b76": {
                            title: "MP3",
                            ext: "mp3",
                            params: {
                                "b:a": "128k",
                                "c:a": "mp3",
                                f: "mp3"
                            },
                            extra: null,
                            audioonly: !0,
                            readonly: !0
                        },
                        "05cb6b27-9167-4d83-833d-218a107d0376": {
                            title: "MP3 HQ",
                            ext: "mp3",
                            params: {
                                "b:a": "256k",
                                "c:a": "mp3",
                                f: "mp3"
                            },
                            extra: null,
                            audioonly: !0,
                            readonly: !0
                        },
                        "69397f64-54f2-4ee4-b47a-b4fc42ee2ec1": {
                            title: "MP4 500Kb",
                            ext: "mp4",
                            params: {
                                "c:v": "mpeg4",
                                "c:a": "aac",
                                f: "mp4",
                                "b:v": "500k",
                                "b:a": "128k",
                                ac: 2
                            },
                            extra: "-mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300",
                            audioonly: !1,
                            readonly: !0
                        },
                        "16044db3-3b75-4155-b549-c0ba19c18887": {
                            title: "MP4 1Mb",
                            ext: "mp4",
                            params: {
                                "c:v": "mpeg4",
                                "c:a": "aac",
                                f: "mp4",
                                "b:v": "1000k",
                                "b:a": "128k",
                                ac: 2
                            },
                            extra: "-mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300",
                            audioonly: !1,
                            readonly: !0
                        },
                        "b5535083-bf16-4ae0-a21f-7c637ce0617f": {
                            title: "MP4 2Mb",
                            ext: "mp4",
                            params: {
                                "c:v": "mpeg4",
                                "c:a": "aac",
                                f: "mp4",
                                "b:v": "2000k",
                                "b:a": "128k",
                                ac: 2
                            },
                            extra: "-mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300",
                            audioonly: !1,
                            readonly: !0
                        },
                        "dfbed97f-46c9-4db8-b5d1-4d19901bc236": {
                            title: "MP4 4Mb",
                            ext: "mp4",
                            params: {
                                "c:v": "mpeg4",
                                "c:a": "aac",
                                f: "mp4",
                                "b:v": "4000k",
                                "b:a": "128k",
                                ac: 2
                            },
                            extra: "-mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300",
                            audioonly: !1,
                            readonly: !0
                        },
                        "912806c1-6c43-44ad-ac6e-05f105bade55": {
                            title: "iPhone",
                            ext: "m4v",
                            params: {
                                "c:v": "mpeg4",
                                "c:a": "aac",
                                s: "480x320",
                                "b:v": "800k",
                                f: "mp4",
                                r: "24",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "2416dcbf-146d-4ca4-b948-f6f702fb043c": {
                            title: "iPod",
                            ext: "m4v",
                            params: {
                                "c:v": "mpeg4",
                                "c:a": "aac",
                                s: "320x240",
                                "b:v": "500k",
                                f: "mp4",
                                r: "24",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "42fb9cf9-94f9-45c1-954f-1c5879f3d372": {
                            title: "Galaxy Tab",
                            ext: "mp4",
                            params: {
                                "c:a": "aac",
                                "b:a": "160k",
                                ac: "2",
                                "c:v": "h264",
                                f: "mp4"
                            },
                            extra: "-crf 22",
                            audioonly: !1,
                            readonly: !0
                        },
                        "edf545c2-88fc-4354-b91d-83e2f31d3c14": {
                            title: "MOV (QuickTime player)",
                            ext: "mov",
                            params: {
                                f: "mov",
                                "c:v": "h264",
                                preset: "fast",
                                "profile:v": "baseline",
                                "c:a": "aac",
                                "b:a": "128k"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "f31ac68e-db3b-4b17-95d7-04456cbc3c26": {
                            title: "Mobile 3GP (Qcif)",
                            ext: "3gp",
                            params: {
                                f: "3gp",
                                "c:v": "h263",
                                "c:a": "aac",
                                "b:a": "12k",
                                s: "176x144",
                                "b:v": "64k",
                                ar: "8000",
                                r: "24"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "85cd71a0-fb61-45a4-9fed-6f2e6e405bc3": {
                            title: "MPEG-2 DVD (PAL)",
                            ext: "mpeg",
                            params: {
                                f: "mpeg2video",
                                target: "pal-dvd"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        },
                        "47b9b2eb-8fd4-4e10-8993-f7d467ed1928": {
                            title: "MPEG-2 DVD (NTSC)",
                            ext: "mpeg",
                            params: {
                                f: "mpeg2video",
                                target: "ntsc-dvd"
                            },
                            extra: null,
                            audioonly: !1,
                            readonly: !0
                        }
                    },
                    m = {
                        240: {
                            x: 7,
                            y: 7,
                            qr: "R0lGODlhHwAfAKEAAAAAAP///wAAAAAAACH5BAEKAAIALAAAAAAfAB8AAAKejI+pi+DvwksTQgNUZk1jWUlHFWXXZoYhWLLmd7KiK8a1etFpO3cgDKRgLLwXypXj/DY/JTLom9lKpMho0utRt0xatjt8eWDg6PF62nXX2+xIar3CSeG4ln4UL4dvyzOed6Y22CTVtNbBpEjkp5T29gHJyIY4eBYFGanVt4KGBsjZQkk1Ryba8IhipXbjN4nlVqPpc4nUKWua5LSrUAAAOw=="
                        },
                        481: {
                            x: 10,
                            y: 10,
                            qr: "R0lGODlhXQBdAKEAAAAAAP///wAAAAAAACH5BAEKAAIALAAAAABdAF0AAAL+jI+py+0Po5y02ouz3gr4D4bih4iWeYzquqxuWMIVarw20N5uDJ5yrWN1PD4cJ0KaEYdGSvL4eE6kqaXTCm1Qkdhqc9rNMou5787LK4940rYQ+EV713K4mhbAu1Xp8VyPtZX3NthTV0hn5weXWMgWGIYIaEYouNd4CRnHOPn447go+ZlJuSlqqMg5GkmKCqral3CWerpX5unaive3WorJqvnrS8ibG9wZOvta+1ls66x5q2Rq+YT6fEUdGQWsbXUdPS2dPc42dyjI7U0Wes6Ivu1QHSTsjmivS39Hjw3+bX2MXzeB/QD+O4iLnxh54bTEWwhxWbpDESvSmkjLYkX+bAwwitFRziOTTa6gldrHJ9lDNSNRcjRZ8uLKlrLm1RtpMCUzUy4HMnyY6IVJCcrssXT4Q6iqmB2JvUPKE8LNC/NwaZzKruawmRqwilPZbKFXcmCZYmBq8yfIpQm5mmv3tGmQYm2jyl1H8+4NukO52N3pKVZWiU6X9dwq+KtMnYlx9npJdnEjio6NWc7AsajUnIdbwdwFV10TtDasLtWsddpom5VoKPypNiiye3FjYp2s1/Vsf6vD3AbdmW8q3kbZwgsb3PhwznGFE0Z+XOlO3lXzksZ7Wndp5Y1pe++eL+le7pS9Xye6ErLhygQZw36fMb7W9tLV2S/vkf54i7PpcK9PDd9Gb/iXX3puYTYgcAX+JdIG/SloIIMDtRcbcMfVRaB+dtUHFoYQ6lchdIt5iFyDAIIBVIRHyTWYXxuqmFc56P0VY2jgYWRid2oFKGNzZc0m2odrzRegMiai9hqRSNWlI3sTbqcki32NCNWDvTD545XlfXbQZVMKZmRDneWUWWHhgZkilL+5JxmQFHIpUJGtbTXkdG9Kmd1JcE6ioYj/3bfcgVGiSaN8gL5i1mbdNHkiirEkKlpIbuV4oY8oYvenk3NtWdsW1OmjaZ04MveWnWZqyChxUdKnUauuvgprrLLOSqtGBQAAOw=="
                        },
                        721: {
                            x: 15,
                            y: 15,
                            qr: "R0lGODlhugC6AKEAAAAAAP///wAAAAAAACH5BAEKAAIALAAAAAC6ALoAAAL+jI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTNe2CeT6zvf+D9QtgjscMTc8Kpc/CfMJTR6NUwX0qnRit01rtrSUcsfICPkMEAepRDX6qn1v3V1S2CvHxvNRfBv8lcCnZzbYJxg4codoyLQnBMhz0+HDBglRGVk0uZFp1/NY9inJqeGpCFpIOrpZinEqAtsgC0LrGpqmeflg69F7y5vKuuvwy2EMzIBsKhy8ijDnR2ywHBDN+Ic9bU2oLercCt0t/nxQfU2enb6Nbt5c/E497l4uX688b69OH87/RIeLTiJ/3wjmwpSPWxVvB9c5klZwVjyFhxw2NIhroL7+NRAvbnzIMGBHjRTZTRS40CKtlQnPTWz30WPJiPjgjHyJ02ZIVf1ikmSZEiPCljkrCh3ac+YiiyjvLeMC8Oa+mCIZ/izKcSc4o1SlZmXK06ROsFax0vT69ajKoF2RiuWqtGnSpzDjoq1DdivNukCn2tULMq/Bq/eilvXbN21beESdNk5qeC1is2PC/usoF0hkxnMfvz1jObDawWz/As6Md7TPundTLyY9uTDm1pRRi645VrVSwpBnH1ZsGnZu3HAT3w4u0fFw5LtZ/wZufKnuzSFcyvZGnXm1Vyd9o7ouOPnnvbV1nfUeq7tWvdmjWzq/Pj346RDblx8GP3wt9fr+3dhX3hsl/NEn4HyviUcegOPJ9IF1AR4z4IHEyeQeetVF2FwjFcb30X+daUiSbSB6OB6JCYLol4gamkihgicuiCKGMZa2oUMsqthIRjMuaOKNtK0Y2o48WjiTj8/FqKOQRtpIZHRC5pfMkhRsF2WV+lHZn5VaSjkBllsmU+OUGH5ZZZhdjkkmmPdV4GWapZhZlZs/ltYkmhY4uA2OQ7pFRmjZNShjoMvh2SefLRqYAaEMzrknYFAZ+qcvgj4onGZE6glcnYhyh6iivOXJaKYcMgdop5NeNiqmlp52qaSmvlrcqatKBp1zd9qaJasXSLdabMtx+murut5Ko2eoQpj+UKTDsvnpcY8WGGyqkAJb66C4ipmssJyBaqiTxWZLbLS5bgslgrNW6lqGKe56rYR+ttDsovjJe2WIVsYLr6yHrlvmt5ReCCuMdk6CLwuK2ianui+Wu1/AC6erZcErHNxawhKrQPGR9MJAaJzM5lhxqBt3IqvH2BqCsMgndPwuuyBrrOrKJbcc7iApazzwx/8WSbPONoeM86bUMqxvokDCrPJ7I/M8rcsoA03ruSQLHfSHvnL7ndUP8zrh0jFDTWDWJYq8dJu9Vns1w6VqvS9cyjrqL9pYy8e2qmXnfLbUCj+cr4t2m3ymtXGr7WrdZMtcj6cwcT3i0F47ri26tzH+fnTNhDs9jeL5UP6y5Y9jHpHmw3H+NOifex553seq3jYfRlONurRVT0X6z6Yj3pPoXNXuOuSO88p7662/Lbl/YE89+sW6rUluupSZ3TXwykvIvLnOuwjx7xoF32Pqew9/PDObT68d9kkajnThu5NvHPGsG58+tOsPLjz84xZfX/iAy2G3tztnHLYAUu8N/Xuew+6WuKL1jHXxelbUEPi/eACweQVkW2XuAjjqTNB6FRSYsTbYNe+BkH9k89/c3GUhELKILyQ8UgZTqED1XW55MfTZk0CzthNyKYSmu6EDkbUzTUUQdgz0IZ18FzvZ3S96RhSXDU83wANqr4mrQ57+DoWoQ7NREYcyhGL5pNjDLR4nYizk1xIn9LogvqmMolIiudJ4xTV+sIY2hOMMb/BD1b2QT3b0Yg3yuLc9sgeJaQKkvdxorj5ajI16cx8aCdm8G/bxkMuKYungpqQptrGSX7xkJJ80ySM2rV6d+2Qmw2hGCmKxd90CpSYbucBO2g6TOwplKjm4yjycb0a23KQqR9W4UdItjqBT5NAE+Uux3fFkyySVZZCJy3nhrpnQoyMzmxnLhhHzjPZL5hOVNk0/VhOMtzOPH0NhzGJmc53yw+YzIXlNaPJwdnzL4RhLKEpu5pJ8vYQlPe+JyHk+EE79zN4/uebIFdJPm1VkIhr+5KnQtJ0zaa18KDs5yUh3/nOXFxTmNYXHPrx9r4Fm+VssM1q/VzqrWSb16EjN570Obq2khzvpBxOKTwu2r6YuRSlOD2pAv/EUozeNaU49iL6BjrCoAZUpSGl60P0xVakEPCov4flSCzo1eGI0KFEFp9PKSa6r6aSqVq260q7m05tFPCtQ56hWIlqyoSac5VjFWNa7utWsXORrE/MKsL0KkKFEg2kXwZnLvg2xUUDc5kd3iFg+GtWl9kxpUw/7zstSFrPAFClWu4nCckozsQabmWY5WVnImhOdk6XlVU9LEEfWlZWuRVJrYyvCoHrSepK8bYdyi71g1hZFP2VSQGc4q8sguRK2vz2ubu3K2+XqEzvADWspo3vK6Rr3jMhtYcK+C97wine85C2vec+L3vSqd73sbe96CwAAOw=="
                        },
                        1081: {
                            x: 25,
                            y: 25,
                            qr: "R0lGODlhNgE2AaEAAAAAAP///wAAAAAAACH5BAEKAAIALAAAAAA2ATYBAAL+jI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTNf2jef6zvf+DwwKh8Si8YhMKpfMpvMJjUqn1KrVAshqt9yu9wsOi8dZCDncO4PN6rb7Td7A5/Q5m56u3+v8/lfuFygIsAeXh/cwqMgHuOjoVvh2aJf4aHnWeKn5V4m4w9i5KbqVOToaCckD6mDaSqjhaoraNmkYGntZivs4q1YrebvrqCus2IupqhdcPEjMHHgcl+zZ8MwLa70YPfabypptjL3WTXr1IuaDLs45zW7Oot7u5ayVjvbeEv95n6Gv448PBUAcAykUtHEw4IiENBhGszdOYQqHMih+i7gPo0T+ExZhdGTw0UXIjRhGwuNXEuU/lSQXsiT40mBMhDNbejC5AucBnRNr2uTAU6DPh+Tq/SQR9ERQh7mWlbvo7oImp76olrHah2i/oU2hzsNKD5pXYGO5bFNW9mlKjQu6VuM6FGvWtFXpXrU7F2/YV3hXvWWrYKnbtmj/RiV8Te+6r30Lg4QLWKYluUwhDwO7uMtZao8jIxA8ubFlxoa1Yd7qeWdoxazXJu5MqTVs07KxjA5XO3Bc0Zx19/Z9GDjt0pozm6X8O0Hl1BGm8g6u/Haz065Jz8ZN/Hhuya8Rx86u9vr07ROWXyavmvnm5Om/iw/vffh7o6ihf14Nnn5+ZPv+71bHPp9/AebV317r2YKeAX7FZ51t3TGoXX8LQghfBebJR6F+A3LT114TClfcc2QVaFyF0T0IYoQbMsRhfRimaOJ9KJ5onwQXApihgDlKI6GHju144IgDGoice0AWWWMALf43HokieuMkkzG29yKNDcJY14Y+skdliBL+iKWOFkoniFZf+rSkVPgNiaSQR7ro5YpcKkhmljvyJaVYCSq55pt5imkljmHiKSdzaTp4XpSFIqgoolcG2iSbT6o4KKGOCgrplHTOKGOSZi7q5qBBxlnplkaK2iaUkv5paaVgZqrhnYNdGqmfoKpqK62a8slpl5S62ueYvW7q6ayowvn+K6x+fCprsNxV2emj0WI6bbLl1bksdbfWKmyiq25rZ67dSutrmdo2O+yotJwL7KnKtvosqe0y+i263uqqZ6PzhvoukZOGe+yZ6aa67p7G9ltirPviGjC4+aqZ7o3cIozsrgdXa7EwptIr7rsfUsyqugCD7HC2FT/cMcYoN4wvoCqbrK/HF9vo7Mvw2vyxzf5m5KlcO9fLctA3gNZzDEQXHbPO7JKcw9HyDr1bjwm7XK7CTAtNk3pLZ420llPfXDXVxJI79tNck80sTFp3+HXaV799ttk0r+1R1F6fLDavxWILdkN2p1yD03I3x7fbSsszuOGB/11q21uH3bfeaFf+xDjci9N9d8iPl20t5BB1DbjfmN/5c+iSJ54UR5UfzvPkbOMducQZj57P6p4XlbfPjhtcOO6REwxO77yrJ7vVOAdfMKvIJw+05ZzPvjfxy8MM8fSvst489qYXb/3I13Y/5+mdP2/87cNHDz7H46bP/PbCJ22+wOyPLzL7ipOfN/ex8z0/6Av3fz/xQQ9t+qtf+jbWv/y9L3vxYyD+Eug6+UEwgAXcnADL90AI0g949rNgBc9HQA3yR3kiVKD0Toi+EKKwhLQ7CuHQlDqpuXCGqoOh7V7YQhrqEIdJOhQIfLjDIMKub0C8SQyFGMQiZrCGOUSiE5f4Opfc8IlCVOL+BTHoISpqMV6Ds2IWtwhG3clQik0MIw2t6MWv/c6MT0TjEUnHxjhCjohvbJwc/7UyLg4wcZQr4xVzh0cewY+Hrigd1i7nP48hkGHOMyCBqqfCROZkimkMpBcFJ4vdDXIGmNwVBwXpQEfmbG4p5KPRKNlJ6pnuk49s2f78qBRUyvJehyRkKwzZyD5Kco6LbF8tRXk9UkYygueY5S4zGMwH9tJcsAugLokJxyEm84/LzOP3SrnBU/qxkhJkpPbWd0tNhvKZphwjJKnVQFe6sWbInCYToSVMaIIzmyS04bCcU7eBRVEE3DynN9s5J3wWU5/mDEE/1cmimf2xlSchaOb++FlHmdmTlsdzZwkUuo13ylOP3gOou+Y4SqE4NJpkPKYjWbnOkRqUnbakpz9ditCJwnOhIY2lShV50YhW1FAKFahIWJrRnE6RlShNKFA1OrFrmlSMJeUXSJPp09rdFGdIhek8O0pTqGK0qtaMZzlf6snWOTWdQzjoL2taxWnqVAVmbSRak6jWoWpzm/xbqxVqaleR0nWFY3UiXuWaz71i05da/Css52pS/bX1KIZdKmI3+k2JHlYijYUsIhNbV8AqpLJfbVoqX/nRNsZ1sgMV7DD7ikTOWhVqxkSdRaOwWLJyFItqLAZRi0paiLZ2tVQV6vRua8m8frGZauSqNYD+203CXtayyiyub5eH3JLFVq+YNS1zgclS1TJzJbsNa0F/2L3o2gu1j+3sL5mq299aEruORcpnh9vU4In3f1glp3lzuc+VWm++klVf3O4bWaVeF7f3HC1Flwtg2Qo4wS3NBn936l8E81bBXp1wg4+7XgIPWJzpnWlWearZn/KVflvdUzVBOd4uCte4oD1tUitM29qGL6TT5eQCU6xcGANSmjPWaoilOlgslniTJ76kj3MLZBfv8cUXXiOPQ+tRstXYvkuuclebbNMDPxXESG5okE2oZSyzWLtYnXJ5rQxmD1uwyIo1apcnOWI0M3S2Tq4nlD/cwxU/17VxvrIzZXz+Zxrr2b19TjM6bSxTBuN5pmR+RpY3TGguVxcXD95yLB6t6DGjN6YF7rF692xhTKu4u6fIcHCPymHWSvq6tjV1cksNau+KeNWKbvWrhbxfFvuOjqQWRaWjHM5Yx7i0ee71Jn696GBHGtLxHV+NbS1d9qo51f919mej2l8lY1jYO551sa2rbBzLGdrNHra3pXztnhoYeaKWNaeh3OgW8/nLdRY3rguNYjh3OtA3PquxjYxvM9sZ3uuet7YNHep4nzSwYVa4B/tNZ4DTW+BgvbLDQWjwWhecxK/tMKM3LmcK/vuDt+62fvcdYQo3F+MJB/m978xtVWY74xzH959LLu/+msN82R/3NL1FDu4crxznZ96uvREe8ocHvb5t/nHEBe3zg+c86U/+ObKpDeGeU/rqJs4vTokLdpxDfeuurni97fhQV+Iy64cm99HPjt/v9jbsZpd2X91OX3Ov0utz1xzd3z72S3OdyHznJdZ1PHVYSzfez/73pq9qcn/vQsOofSvh5W74v0M+8b4+ddTJ23XMQ1HzdLb7P5PNZNS7+7yFH73fX7/4Iwu+7Gp3rujFTHq2MxnvM2/50jHY+N9z/tie53f41tz6ha69J28OsMqDynMWFrLokd/8zdMOXulPnvpwLz3yb89m7R8/5u1+OavzKv7tMzzTSp90e2OfftD+R3/11v8+9j8Q/9kTm/0sp3/fPZ5/2KZvzBZK1zZmAZh6uqaAoXd/RoSAnbd+oQZ07keAefeAp1d+1ed9DEhSAHiB0wZXEdNxe5d7Dvh+mwVUljdOh9cBgwZbKTiCrFeCLeh074BRKkiCsOeBendGMDh+l6eDJ1eB+HCDMRh3QZh9JxgQRfiDKziDQFGD5sCEOweEdbeAbDSFKYd788eD9RdHWSh/ToiE+BeFVwCGGGh/Vkh+cnSGQnd9asiFGmhpE3SATQiHBZhrVUeHMYeDtdd/2zZwexiHc0aGNkd5CVh87FaHVDiGzxdegKZBGUiIJih1pmd07yaI5SZzSVj+iYd4iZsnfZLYh164SY+ohwkkika4gaWYh4GIiouohVBoiIl4aJYIiJo4iTQ4i2KHarz4aYM4iquIh794h57VfJTIf1Xoh8pzhdylhEKYjGJYjKDYjMb4jJwYjTk4jRFXjWpzjYX4jc+3fE3WfYj3OUOIjRKYhstoduW4hc6IjuAYj+LIgtCHi+eYjdCojhzIjurUjao2j7oYjkN3j7ZXkIizj4MoiQtpkDsoh8w3kLIYkcJlgGuIkP7HVmVIkA7pjspnkWKVkAepkP+4YCEpeWFGTdyXi7a4iSWJke8IeBCXkb0Ifyj5ePJIc594hBy5kp7oZzeJjDrnYS4YkyP+p37M+JE12YkRiIi+aJPJN46qN0LSqI9NqZS1CJQCKXUXt39YSYtW2YFVaX61SJQWmG9Pd5TtSJJziIFl2Xtu6JMtiXgd2X5DqZEs+ZBSSXxDtJbA5oZuqXtw+ZU66ZJdWJjDx5VUCYrq9nlC6ZWsKIK7iIsCyJaOaZmEaY5xOXxviIlt53JI95N/KJcbKYOuuHufiZiq6Ig+aHV8yJjGN3GqSZpvOZXuc4x+iZl6uZmaGZVtOJYmaZaBV3mZxYi2iZIkN5mvGYtNV5wnaZeSyZPKOZwBJ5spOZi/SXXAKJ1tSZyxWJdkKZNBmUlE951g6ZzgCZ1iCYHRVp65OYH+zxmbrViUF9iXuomc86l/+ImA9emboNlBTvmA/MmaW2mKVxmAAhqZ8UmMwfmBdCmaK8mc0HWd6Yegx9mdigig+5mU+nmZ2Dk/mql9FQqfBCqfDPqBIoqeCiqhGXqgG2qircmi42mg+VefVOah/umgTPmS1qmM6bijDPmgSRB8MHpuwOmjN7qbRzCkS9mVP4qiHZqXNmqYq3mbnOllTKqY1baj93ml+XikOGqlWjqls+l81OWlOAmlSWoES5qTYyqRAfmmbQqmSMCmaSqlOTqSQUqn77WOX4qnIjmMEalp9QiY9Kijc8qPAHl+6TmAZyqeSBqmEral4WmmNNqnwqimjWjHoTLajiwUqZmZqP/3ovnZmQB0qYf5qR4Zo4rXqSWUqu+pcYlmqqc4qz2KqaUZlrTpaJCYiZlKilmaeTPKqqX6oacKqrYarJs6rIvpqsYKk7iqqaM6fbRarKGKqs6aUpHIq69orccKrK6nrOvZqiL0qu3pptnaq3ekruvKru3qru8Kr/Eqr/NKr/Vqr/eKr/mqr/vKr/3qr/8KsAErsANLsPZaAAA7"
                        }
                    }
            },
            197: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.set = function (e) {
                    a = e, s()
                }, t.outputConfigForHit = function (e) {
                    var t = (e.url || e.videoUrl || e.audioUrl) && e.topUrl;
                    if (!t) return Promise.resolve(null);
                    for (var n = new URL(t).hostname, r = [], i = n.split("."), s = 0; s < i.length - 1; s++) r.push(i.slice(s).join("."));
                    var u = null;
                    if (a.every((function (t) {
                            var n = !0;
                            return t.extension && e.extension !== t.extension && (n = !1), n && t.domain && (n = !r.every((function (e) {
                                return e !== t.domain
                            }))), n && (u = t), !n
                        }))) return Promise.resolve(null);
                    if (!u.convert) return Promise.resolve(null);
                    return o.getOutputConfigs().then((function (e) {
                        return e[u.format] || null
                    }))
                };
                var r = n(3),
                    i = r.browser,
                    o = n(192),
                    a = [];

                function s() {
                    return i.storage.local.set({
                        convrules: a
                    }).catch((function (e) {
                        console.error("Cannot write conversion rules storage")
                    }))
                }
                r.rpc.listen({
                    editConversionRules: function () {
                        r.ui.open("convrules-edit", {
                            type: "tab",
                            url: "content/convrules-edit.html"
                        })
                    },
                    getConversionRules: function () {
                        return a
                    },
                    setConversionRules: function (e) {
                        return a = e, s()
                    }
                }), i.storage.local.get({
                    convrules: []
                }).then((function (e) {
                    a = e.convrules
                })).catch((function (e) {
                    console.error("Cannot read conversion rules storage")
                }))
            },
            241: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = function () {
                        function e(e, t) {
                            for (var n = 0; n < t.length; n++) {
                                var r = t[n];
                                r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                            }
                        }
                        return function (t, n, r) {
                            return n && e(t.prototype, n), r && e(t, r), t
                        }
                    }(),
                    i = function e(t, n, r) {
                        null === t && (t = Function.prototype);
                        var i = Object.getOwnPropertyDescriptor(t, n);
                        if (void 0 === i) {
                            var o = Object.getPrototypeOf(t);
                            return null === o ? void 0 : e(o, n, r)
                        }
                        if ("value" in i) return i.value;
                        var a = i.get;
                        return void 0 !== a ? a.call(r) : void 0
                    };
                var o = n(3),
                    a = n(220),
                    s = a.Chunkset,
                    u = n(32),
                    c = n(78),
                    l = n(7);
                t.DashChunkset = function (e) {
                    function t(e) {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, t);
                        var n = function (e, t) {
                            if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                            return !t || "object" != typeof t && "function" != typeof t ? e : t
                        }(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                        return e.descrPrefix = o._("dash_streaming"), n.endsWhenNoMoreSegment = !0, n
                    }
                    return function (e, t) {
                        if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                        e.prototype = Object.create(t && t.prototype, {
                            constructor: {
                                value: e,
                                enumerable: !1,
                                writable: !0,
                                configurable: !0
                            }
                        }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                    }(t, e), r(t, [{
                        key: "processChunkData",
                        value: function (e, t) {
                            this.chunkIndex = (this.chunkIndex || 0) + 1;
                            var n = 0,
                                r = c.getTags("moof", e);
                            if (r.length < 1) return t(new Error("No moof in fragment"));
                            var i = r[0].offset,
                                o = c.getTags("mfhd", r[0].data);
                            if (o.length < 1) return t(new Error("No mfhd in fragment"));
                            var s = u.ReadInt32(o[0].data, 4);
                            this.seqNum = s;
                            var l = 0;
                            for (var d in this.esis) this.esis[d].sampleGroups = [];
                            for (var f = c.getTags("traf", r[0].data), p = 0, h = f.length; p < h; p++) {
                                var g = f[p],
                                    m = c.getTags("tfhd", g.data);
                                if (m.length < 1) return t(new Error("No tfhd in track fragment"));
                                var v = m[0].data,
                                    y = u.ReadInt32(v, 4);
                                y >= this.nextTrackId && (this.nextTrackId = y + 1);
                                var b = this.esis[y] = this.esis[y] || {
                                    trackId: y,
                                    dataOffsets: [],
                                    dataSizes: [],
                                    keyFrames: [],
                                    sampleGroups: [],
                                    sampleCount: 0,
                                    chunkNumber: 0,
                                    sampleSizes: [],
                                    duration: 0,
                                    streamType: this.init.streamTypes[y]
                                };
                                if (this.mpd.codecs) {
                                    var w = this.mpd.codecs.split(",");
                                    if (y <= w.length)
                                        for (var k in a.Codecs) {
                                            var x = a.Codecs[k];
                                            0 == w[y - 1].indexOf(x.strTag) && (b.codecId = k, b.codec = x, b.streamType = x.type)
                                        }
                                }
                                var A = u.ReadInt24(v, 1),
                                    _ = (1 & A) >>> 0,
                                    O = (2 & A) >>> 1 >>> 0,
                                    I = (8 & A) >>> 3 >>> 0,
                                    P = (16 & A) >>> 4 >>> 0,
                                    C = (32 & A) >>> 5 >>> 0;
                                b.durationIsEmpty = (65536 & A) >>> 16 >>> 0;
                                var S = (131072 & A) >>> 17 >>> 0,
                                    j = 8;
                                _ ? (l = u.ReadInt64(v, j) - this.globalOffset, j += 8) : (S || 0 == l) && (l = i), O && (b.sampleDescriptionIndex = u.ReadInt32(v, j), j += 4), I && (b.defaultSampleDuration = u.ReadInt32(v, j), j += 4), P && (b.defaultSampleSize = u.ReadInt32(v, j), j += 4), C && (b.defaultSampleFlags = u.ReadInt32(v, j), j += 4);
                                for (var E = c.getTags("trun", g.data), D = 0, T = E.length; D < T; D++) {
                                    var R = E[D],
                                        q = {
                                            s: 0,
                                            o: 0,
                                            d: 0
                                        };
                                    b.sampleGroups.push(q);
                                    var M = R.data,
                                        F = u.ReadInt24(M, 1),
                                        W = 1 & F,
                                        N = (4 & F) >>> 2 >>> 0,
                                        U = (256 & F) >>> 8 >>> 0,
                                        z = (512 & F) >>> 9 >>> 0,
                                        L = (1024 & F) >>> 10 >>> 0,
                                        V = (2048 & F) >>> 11 >>> 0,
                                        H = u.ReadInt32(M, 4),
                                        K = 8;
                                    if (W) {
                                        var B = u.ReadInt32(M, K);
                                        K += 4, q.o = l + B
                                    } else 0 == l && (q.o = l);
                                    if (N) {
                                        u.ReadInt32(M, K);
                                        K += 4
                                    }
                                    for (var Z = 0; Z < H; Z++) {
                                        var Y = {};
                                        U ? (Y.d = u.ReadInt32(M, K), K += 4) : Y.d = b.defaultSampleDuration, z ? (Y.s = u.ReadInt32(M, K), K += 4) : Y.s = b.defaultSampleSize, L ? (Y.f = u.ReadInt32(M, K), K += 4) : Y.f = b.defaultSampleFlags, 33554432 & Y.f && b.keyFrames.push(b.sampleCount + Z), V && (Y.C = u.ReadInt32(M, K), K += 4), q.s += Y.s, q.d += Y.d, b.sampleSizes.push(Y.s), b.duration += Y.s, b.stts = b.stts || [], 0 == b.stts.length || b.stts[b.stts.length - 1].d != Y.d ? b.stts.push({
                                            c: 1,
                                            d: Y.d
                                        }) : b.stts[b.stts.length - 1].c++
                                    }
                                    q.c = H, l = q.o + q.s, b.sampleCount += H, b.stsc = b.stsc || [], 0 != b.stsc.length && b.stsc[b.stsc.length - 1].samples_per_chunk == H || b.stsc.push({
                                        first_chunk: b.chunkNumber,
                                        samples_per_chunk: H,
                                        sample_description_index: 0
                                    }), b.chunkNumber++
                                }
                            }
                            this.globalOffset += e.length;
                            var J = [];
                            for (var X in this.esis)
                                for (var G = this.esis[X], Q = 0; Q < G.sampleGroups.length; Q++) {
                                    var $ = G.sampleGroups[Q];
                                    J.push(e.subarray($.o, $.o + $.s)), G.dataOffsets.push({
                                        b: this.chunkIndex - 1,
                                        o: n
                                    }), G.dataSizes.push($.s), this.dataOffset += $.s, n += $.s
                                }
                            t.call(this, null, J)
                        }
                    }, {
                        key: "getTrackDuration",
                        value: function (e) {
                            return this.getTotalDuration()
                        }
                    }, {
                        key: "getTotalDuration",
                        value: function () {
                            return Math.round(1e3 * this.mpd.duration)
                        }
                    }, {
                        key: "finalize",
                        value: function (e, n) {
                            var r = this,
                                o = [];
                            for (var a in this.esis) {
                                var s = this.esis[a];
                                o.push(s)
                            }
                            var u = this,
                                l = function () {
                                    for (var e, n = arguments.length, o = Array(n), a = 0; a < n; a++) o[a] = arguments[a];
                                    (e = i(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "finalize", r)).call.apply(e, [r].concat(o))
                                }.bind(this);
                            this.waitForWrittenData((function () {
                                c.finalize(u, o, u.downloadTarget, (function (e) {
                                    l(e, n)
                                }))
                            }))
                        }
                    }, {
                        key: "handleInitSegment",
                        value: function (e) {
                            if (this.mpd = e, this.init = {}, this.segmentsCount = e.segments.length, e.init_segment) try {
                                var t = void 0;
                                t = "string" == typeof e.init_segment ? l.toByteArray(e.init_segment) : e.init_segment, this.globalOffset = t.length;
                                var n = c.getTags("ftyp", t);
                                this.init.ftyp = n[0].data;
                                var r = c.getTags("moov", t);
                                this.init.stsd = {}, this.init.tkhd = {}, this.init.vmhd = {}, this.init.smhd = {}, this.init.edts = {}, this.init.hdlr = {}, this.init.streamTypes = {}, this.init.mdhd = {}, this.init.dinf = {}, this.init.edts = {}, this.timeScale = {};
                                var i = c.getTags("mvhd", r[0].data);
                                this.init.mvhd = i[0].data, this.init.timescale = u.ReadInt32(i[0].data, 12), this.init.duration = u.ReadInt32(i[0].data, 16), 0 == this.init.duration && (this.init.duration = Math.round((e.duration || 0) * this.init.timescale), u.WriteInt32(i[0].data, 16, this.init.duration));
                                var o = c.getTags("iods", r[0].data);
                                o.length > 0 && (this.init.iods = o[0].data);
                                for (var a = c.getTags("trak", r[0].data), s = 0; s < a.length; s++) {
                                    var d = a[s],
                                        f = c.getTags("tkhd", d.data),
                                        p = u.ReadInt32(f[0].data, 12),
                                        h = f[0].data;
                                    this.init.tkhd[p] = h;
                                    var g = u.ReadInt32(h, 20);
                                    0 == g && (g = this.init.duration, u.WriteInt32(h, 20, g));
                                    var m = c.getTags("edts", d.data);
                                    m.length > 0 && (this.init.edts[p] = m[0].data);
                                    var v = c.getTags("mdia", d.data),
                                        y = c.getTags("hdlr", v[0].data);
                                    this.init.hdlr[p] = y[0].data;
                                    var b = u.ReadInt32(this.init.hdlr[p], 8);
                                    this.init.streamTypes[p] = 1936684398 === b ? "audio" : 1986618469 === b ? "video" : void 0;
                                    var w = c.getTags("dinf", v[0].data);
                                    w.length > 0 && (this.init.dinf[p] = w[0].data);
                                    var k = c.getTags("minf", v[0].data),
                                        x = c.getTags("mdhd", v[0].data)[0].data;
                                    this.init.mdhd[p] = x;
                                    var A = u.ReadInt32(x, 16),
                                        _ = u.ReadInt32(x, 12);
                                    this.timeScale[p] = _, 0 == A && (A = Math.round(this.init.duration * _ / this.init.timescale), u.WriteInt32(x, 16, A));
                                    var O = c.getTags("vmhd", k[0].data);
                                    O.length > 0 && (this.init.vmhd[p] = O[0].data);
                                    var I = c.getTags("smhd", k[0].data);
                                    I.length > 0 && (this.init.smhd[p] = I[0].data);
                                    var P = c.getTags("stbl", k[0].data),
                                        C = c.getTags("stsd", P[0].data);
                                    this.init.stsd[p] = C[0].data
                                }
                            } catch (e) {
                                console.warn("Error decoding DASH init segment")
                            }
                        }
                    }, {
                        key: "download",
                        value: function (e, t, n, r, i) {
                            var o = this;
                            this.action = e, this.downloadTarget = t[0].fileName;
                            var a = e.hit._mpd || e.hit.audioMpd,
                                s = new URL(e.hit._mpdCommonBaseUrl, e.hit.url || e.hit.audioUrl).href;
                            s = new URL(a.base_url, s).href, o.downloadFile(this.downloadTarget, a, s, n, r, i), e.abortChunked = function () {
                                o.actionAbortFn()
                            }
                        }
                    }, {
                        key: "downloadFile",
                        value: function (e, t, n, r, i, o) {
                            var a = this;
                            this.aborted = !1, this.successFn = r, this.errorFn = i, this.progressFn = o, this.downloadTarget = e, this.dataOffset = 0, this.nextTrackId = 1, this.chunks = [], this.dataOffset = 0, this.globalOffset = 0, this.esis = {}, this.seqNum = 0, this.processedSegmentsCount = 0, "string" == typeof t.init_segment && (t.init_segment = l.toByteArray(t.init_segment)), this.handleInitSegment(t), this.mpd.segments.forEach((function (e) {
                                var t = new URL(e.url, n).href;
                                a.chunks.push({
                                    url: t,
                                    index: a.chunks.length
                                })
                            })), c.writeFileHeader(this, (function (e) {
                                e ? i(e) : (a.recording = !0, a.handle())
                            }))
                        }
                    }]), t
                }(s)
            },
            213: e => {
                "use strict";
                e.exports = [{
                    name: "networkProbe",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "titleMode",
                    type: "choice",
                    defaultValue: "right",
                    choices: ["right", "left", "multiline"]
                }, {
                    name: "iconActivation",
                    type: "choice",
                    defaultValue: "currenttab",
                    choices: ["currenttab", "anytab"]
                }, {
                    name: "iconBadge",
                    type: "choice",
                    defaultValue: "tasks",
                    choices: ["none", "tasks", "activetab", "anytab", "pinned", "mixed"]
                }, {
                    name: "hitsGotoTab",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "default-action-0",
                    type: "string",
                    defaultValue: "quickdownload",
                    hidden: !0
                }, {
                    name: "default-action-1",
                    type: "string",
                    defaultValue: "openlocalfile",
                    hidden: !0
                }, {
                    name: "default-action-2",
                    type: "string",
                    defaultValue: "abort",
                    hidden: !0
                }, {
                    name: "smartnamerFnameSpaces",
                    type: "choice",
                    defaultValue: "keep",
                    choices: ["keep", "remove", "hyphen", "underscore"]
                }, {
                    name: "smartnamerFnameMaxlen",
                    type: "integer",
                    defaultValue: 64,
                    minimum: 12,
                    maximum: 256
                }, {
                    name: "downloadControlledMax",
                    type: "integer",
                    defaultValue: 6,
                    minimum: 0
                }, {
                    name: "downloadStreamControlledMax",
                    type: "integer",
                    defaultValue: 6,
                    minimum: 0
                }, {
                    name: "autoPin",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "mediaExtensions",
                    type: "string",
                    defaultValue: "flv|ram|mpg|mpeg|avi|rm|wmv|mov|asf|mp3|rar|movie|divx|rbs|mp4|mpeg4"
                }, {
                    name: "dashHideM4s",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "mpegtsHideTs",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "orphanExpiration",
                    type: "integer",
                    defaultValue: 60,
                    minimum: 0
                }, {
                    name: "chunksEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "hlsEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "dashEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "dashOnAdp",
                    type: "choice",
                    defaultValue: "audio_video",
                    choices: ["audio", "video", "audio_video"]
                }, {
                    name: "hlsDownloadAsM2ts",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "networkFilterOut",
                    type: "string",
                    defaultValue: "/frag\\\\([0-9]+\\\\)/|[&\\\\?]range=[0-9]+-[0-9]+|/silverlight/"
                }, {
                    name: "mediaweightThreshold",
                    type: "integer",
                    defaultValue: 2097152
                }, {
                    name: "mediaweightMinSize",
                    type: "integer",
                    defaultValue: 8192
                }, {
                    name: "tbvwsEnabled",
                    type: "boolean",
                    defaultValue: !0,
                    hidden: !0
                }, {
                    name: "ignoreProtectedVariants",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "qualitiesMaxVariants",
                    type: "integer",
                    defaultValue: 6
                }, {
                    name: "adpHide",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "adaptativeCount",
                    type: "integer",
                    defaultValue: 4,
                    hidden: !0
                }, {
                    name: "converterThreads",
                    type: "string",
                    defaultValue: "auto"
                }, {
                    name: "converterAggregTuneH264",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "notifyReady",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "noPrivateNotification",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "avplayEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "blacklistEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "chunksConcurrentDownloads",
                    type: "integer",
                    defaultValue: 4
                }, {
                    name: "chunksPrefetchCount",
                    type: "integer",
                    defaultValue: 4
                }, {
                    name: "downloadRetries",
                    type: "integer",
                    defaultValue: 3
                }, {
                    name: "downloadRetryDelay",
                    type: "integer",
                    defaultValue: 1e3
                }, {
                    name: "mpegtsSaveRaw",
                    type: "boolean",
                    defaultValue: !1,
                    hidden: !0
                }, {
                    name: "mpegtsSaveRawStreams",
                    type: "boolean",
                    defaultValue: !1,
                    hidden: !0
                }, {
                    name: "mpegtsEndsOnSeenChunk",
                    type: "boolean",
                    defaultValue: !0,
                    hidden: !0
                }, {
                    name: "converterKeepTmpFiles",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "backgroundReduxLogger",
                    type: "boolean",
                    defaultValue: !1,
                    hidden: !0
                }, {
                    name: "dlconvLastOutput",
                    type: "string",
                    defaultValue: "",
                    hidden: !0
                }, {
                    name: "linuxLicense",
                    type: "boolean",
                    defaultValue: !1,
                    hidden: !0
                }, {
                    name: "qrMessageNotAgain",
                    type: "boolean",
                    defaultValue: !1,
                    hidden: !0
                }, {
                    name: "coappShellEnabled",
                    type: "boolean",
                    defaultValue: !1,
                    hidden: !0
                }, {
                    name: "downloadCount",
                    type: "integer",
                    defaultValue: 0,
                    hidden: !0
                }, {
                    name: "donateNotAgainExpire",
                    type: "integer",
                    defaultValue: 0,
                    hidden: !0
                }, {
                    name: "popupHeightLeftOver",
                    type: "integer",
                    defaultValue: 100,
                    hidden: !0
                }, {
                    name: "coappDownloads",
                    type: "choice",
                    defaultValue: "ask",
                    choices: ["ask", "coapp", "browser"]
                }, {
                    name: "lastDownloadDirectory",
                    type: "string",
                    defaultValue: "dwhelper"
                }, {
                    name: "fileDialogType",
                    type: "choice",
                    defaultValue: "tab",
                    choices: ["tab", "panel"]
                }, {
                    name: "alertDialogType",
                    type: "choice",
                    defaultValue: "tab",
                    choices: ["tab", "panel"]
                }, {
                    name: "monitorNetworkRequests",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "chunkedCoappManifestsRequests",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "chunkedCoappDataRequests",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "coappRestartDelay",
                    type: "integer",
                    defaultValue: 1e3
                }, {
                    name: "rememberLastDir",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "coappIdleExit",
                    type: "integer",
                    defaultValue: 6e4
                }, {
                    name: "dialogAutoClose",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "convertControlledMax",
                    type: "integer",
                    defaultValue: 1
                }, {
                    name: "checkCoappOnStartup",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "coappUseProxy",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "downloadCompleteDelay",
                    type: "integer",
                    defaultValue: 1e3
                }, {
                    name: "contentRedirectEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "contextMenuEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "toolsMenuEnabled",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "medialinkExtensions",
                    type: "string",
                    defaultValue: "jpg|jpeg|gif|png|mpg|mpeg|avi|rm|wmv|mov|flv|mp3|mp4"
                }, {
                    name: "medialinkMaxHits",
                    type: "integer",
                    defaultValue: 50
                }, {
                    name: "medialinkMinFilesPerGroup",
                    type: "integer",
                    defaultValue: 6
                }, {
                    name: "medialinkMinImgSize",
                    type: "integer",
                    defaultValue: 80
                }, {
                    name: "medialinkAutoDetect",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "medialinkScanImages",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "medialinkScanLinks",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "bulkEnabled",
                    type: "boolean",
                    defaultValue: !0
                }, {
                    name: "tbvwsGrabDelay",
                    type: "integer",
                    defaultValue: 2e3
                }, {
                    name: "forcedCoappVersion",
                    type: "string",
                    regexp: "^$|^\\d+\\.\\d+\\.\\d+$",
                    defaultValue: ""
                }, {
                    name: "lastHlsDownload",
                    type: "integer",
                    defaultValue: 0,
                    hidden: !0
                }, {
                    name: "galleryNaming",
                    type: "choice",
                    choices: ["type-index", "url", "index-url"],
                    defaultValue: "type-index"
                }, {
                    name: "hlsRememberPrevLiveChunks",
                    type: "boolean",
                    defaultValue: !1
                }, {
                    name: "hlsEndTimeout",
                    type: "integer",
                    defaultValue: 20
                }, {
                    name: "tbvwsExtractionMethod",
                    type: "choice",
                    choices: ["page", "android", "ios", "tvep"],
                    defaultValue: "ios"
                }]
            },
            194: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.dialog = s, t.alert = u, t.fileDialog = c, t.saveAs = function (e, t) {
                    var n = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {};
                    return c(Object.assign({
                        filename: e,
                        directory: t,
                        uniqueFilename: !0,
                        titleText: r._("save_file_as"),
                        noSizeColumn: !1,
                        dirOnly: !1,
                        upDir: !0,
                        editFileInput: !0,
                        readonlyDir: !1,
                        showDir: !0,
                        okText: r._("save"),
                        confirmOverwrite: !0,
                        newDir: !0,
                        createDir: !0
                    }, n))
                }, t.selectDirectory = l, t.selectConvertFiles = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                    return c(Object.assign({
                        directory: e,
                        uniqueFilename: !1,
                        titleText: r._("select_files_to_convert"),
                        noSizeColumn: !1,
                        dirOnly: !1,
                        upDir: !0,
                        readonlyDir: !0,
                        editFileInput: !1,
                        showDir: !1,
                        okText: r._("convert"),
                        confirmOverwrite: !1,
                        newDir: !1,
                        createDir: !1,
                        selectMultiple: !0,
                        outputConfigs: !0
                    }, t))
                }, t.selectMergeVideoFile = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                    return c(Object.assign({
                        directory: e,
                        uniqueFilename: !1,
                        titleText: r._("select_video_file_to_merge"),
                        noSizeColumn: !1,
                        dirOnly: !1,
                        upDir: !0,
                        readonlyDir: !0,
                        editFileInput: !1,
                        showDir: !1,
                        okText: r._("next"),
                        confirmOverwrite: !1,
                        newDir: !1,
                        createDir: !1,
                        selectMultiple: !1,
                        outputConfigs: !1
                    }, t))
                }, t.selectMergeAudioFile = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                    return c(Object.assign({
                        directory: e,
                        uniqueFilename: !1,
                        titleText: r._("select_audio_file_to_merge"),
                        noSizeColumn: !1,
                        dirOnly: !1,
                        upDir: !0,
                        readonlyDir: !0,
                        editFileInput: !1,
                        showDir: !1,
                        okText: r._("next"),
                        confirmOverwrite: !1,
                        newDir: !1,
                        createDir: !1,
                        selectMultiple: !1,
                        outputConfigs: !1
                    }, t))
                };
                var r = n(3),
                    i = r.browser,
                    o = n(49),
                    a = 0;

                function s(e) {
                    var t = Promise.resolve();
                    "tab" === e.type && (t = t.then((function () {
                        return i.tabs.query({
                            active: !0,
                            lastFocusedWindow: !0
                        }).then((function (e) {
                            e.length > 0 && o.setTransientTab("<next-tab>", e[0].id)
                        }))
                    })));
                    var n = "dialog" + ++a;
                    return (t = t.then((function () {
                        r.ui.open(n, e)
                    })).then((function () {
                        return r.wait(n)
                    }))).__dialogName = n, t
                }

                function u(e) {
                    var t = {
                        autoResize: !0
                    };
                    return "tab" == r.prefs.alertDialogType && (t = {
                        bodyClass: "dialog-in-tab",
                        autoResize: !1
                    }), s({
                        url: "content/alert.html",
                        type: r.prefs.alertDialogType,
                        height: e.height || 200,
                        autoClose: r.prefs.dialogAutoClose,
                        initData: Object.assign(t, e)
                    })
                }

                function c(e) {
                    var t = s({
                        type: r.prefs.fileDialogType,
                        url: "content/file-dialog.html",
                        height: 500,
                        width: 750,
                        autoClose: r.prefs.dialogAutoClose,
                        initData: Object.assign({
                            filename: null,
                            directory: null,
                            uniqueFilename: !0,
                            titleText: "",
                            noSizeColumn: !1,
                            dirOnly: !1,
                            upDir: !0,
                            editFileInput: !0,
                            readonlyDir: !1,
                            showDir: !0,
                            okText: "OK",
                            confirmOverwrite: !1,
                            newDir: !1,
                            createDir: !0
                        }, e)
                    });
                    return t.then((function (e) {
                        return r.ui.close(t.__dialogName), e
                    })).catch((function (e) {
                        return r.ui.close(t.__dialogName), null
                    }))
                }

                function l(e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                    return c(Object.assign({
                        directory: e,
                        uniqueFilename: !1,
                        titleText: r._("weh_prefs_label_lastDownloadDirectory"),
                        noSizeColumn: !0,
                        dirOnly: !0,
                        upDir: !0,
                        editFileInput: !1,
                        readonlyDir: !0,
                        showDir: !1,
                        okText: r._("ok"),
                        confirmOverwrite: !1,
                        newDir: !0,
                        createDir: !1
                    }, t))
                }
                r.rpc.listen({
                    alert: u,
                    selectDirectory: l
                })
            },
            238: (e, t, n) => {
                "use strict";
                var r = n(3),
                    i = r.browser,
                    o = n(7),
                    a = n(34),
                    s = 1e3,
                    u = 0,
                    c = {},
                    l = null,
                    d = null;

                function f(e) {
                    return e && e.coappDownload ? a.downloads : d || ((d = {}).cancel = i.downloads.cancel, d.search = i.downloads.search, d.download = function (e) {
                        return e.headers && (e.headers = []), delete e.directory, delete e.proxy, i.downloads.download(e)
                    }, d)
                }

                function p() {
                    var e = Object.keys(c).length;
                    l && 0 == e ? (clearInterval(l), l = null) : !l && e > 0 && (l = setInterval(h, s))
                }

                function h() {
                    var e = function (e) {
                        var t = c[e];
                        f(t.data).search({
                            id: t.downloadId
                        }).then((function (n) {
                            if (n.length > 0) {
                                var i = n[0];
                                if ("in_progress" == i.state) {
                                    var a = Math.floor(100 * i.bytesReceived / i.totalBytes);
                                    a != t.lastProgress && (t.lastProgress = a, t.progress(a)), i.error && (f(t.data).cancel(i.id), t.failure(new o.DetailsError(r._("download_error"), i.error)), delete c[e], p())
                                } else delete c[e], p(), "complete" == i.state ? setTimeout((function () {
                                    t.success(i.filename)
                                }), r.prefs.downloadCompleteDelay) : "Aborted" == i.error ? t.failure(new o.VDHError(r._("download_error"), {
                                    noReport: !0
                                })) : t.failure(new o.DetailsError(r._("download_error"), i.error))
                            } else console.warn("Not found download", e)
                        }))
                    };
                    for (var t in c) e(t)
                }

                function g() {}

                function m(e, t) {
                    p(), e.failure(t)
                }
                t.download = function (e, t, n, a) {
                    var s = ++u,
                        l = {
                            id: s,
                            data: e,
                            success: t || g,
                            failure: n || g,
                            progress: a || g
                        };
                    p(), l.lastProgress = -1;
                    var d = {
                        url: l.data.source.url,
                        conflictAction: "uniquify",
                        filename: l.data.target.filename,
                        directory: l.data.target.directory,
                        saveAs: l.data.target.saveAs || !1,
                        incognito: !!l.data.source.isPrivate
                    };
                    return l.data.source.headers ? d.headers = l.data.source.headers : l.data.source.referrer && (d.headers = [{
                        name: "Referer",
                        value: l.data.source.referrer
                    }]), l.data.proxy && r.prefs.coappUseProxy && (d.proxy = l.data.proxy), Promise.resolve(i.runtime.getBrowserInfo && i.runtime.getBrowserInfo() || null).then((function (e) {
                        return (!e || "Firefox" != e.name || parseInt(e.version) <= 56) && delete d.incognito, f(l.data).download(d)
                    })).then((function (e) {
                        if (!e) return m(l, new Error(r._("aborted"))), void p();
                        l.downloadId = e, c[l.id] = l, p()
                    })).catch((function (e) {
                        m(l, new o.DetailsError(r._("download_error"), e.message))
                    })), s
                }, t.abort = function (e) {
                    c[e] && f(c[e].data).cancel(c[e].downloadId)
                }
            },
            243: (e, t, n) => {
                "use strict";
                var r = n(3),
                    i = n(191);
                t.newDownload = function () {
                    var e = r.prefs.downloadCount;
                    e++, r.prefs.downloadCount = e, e > 0 && e % 100 == 0 && (Math.round(Date.now() / 1e3) < r.prefs.donateNotAgainExpire || i.checkLicense().then((function (e) {
                        e && "accepted" == e.status || r.ui.open("funding", {
                            type: r.prefs.alertDialogType,
                            url: "content/funding.html",
                            height: 550
                        })
                    })))
                }, r.rpc.listen({
                    fundingLater: function () {
                        r.prefs.donateNotAgainExpire = Math.round(Date.now() / 1e3) + 2592e3
                    }
                })
            },
            225: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = n(3),
                    i = r.browser,
                    o = n(12),
                    a = n(7),
                    s = n(226);
                async function u() {
                    var e = await i.tabs.query({
                        active: !0,
                        currentWindow: !0
                    });
                    if (0 === e.length) throw new Error("Can't find current tab");
                    return {
                        tabId: e[0].id
                    }
                }
                async function c(e) {
                    var t = void 0;
                    t = e ? {
                        tabId: e
                    } : await u(), await i.scripting.insertCSS({
                        target: t,
                        css: ".vdh-mask { position: absolute; display: none; background-color: rgba(255,0,0,0.5); z-index: 2147483647; }"
                    }), await a.executeScriptWithGlobal(t, {
                        _$vdhParams: {
                            extensions: r.prefs.medialinkExtensions,
                            maxHits: r.prefs.medialinkMaxHits,
                            minFilesPerGroup: r.prefs.medialinkMinFilesPerGroup,
                            minImgSize: r.prefs.medialinkMinImgSize,
                            scanImages: r.prefs.medialinkScanImages,
                            scanLinks: r.prefs.medialinkScanLinks
                        }
                    }, "/content/gallery-script.js")
                }

                function l(e) {
                    return ".vdh-mask." + e + " { display: block; }"
                }
                t.analyzePage = c, r.rpc.listen({
                    analyzePage: function () {
                        c()
                    },
                    galleryGroups: function (e) {
                        Object.keys(e.groups).forEach((function (t) {
                            var n = e.groups[t],
                                i = "??",
                                u = "??";
                            try {
                                u = new URL(n.baseUrl).hostname
                            } catch (e) {
                                console.warn("Uncaught URL error", e)
                            }
                            switch (n.type) {
                                case "image":
                                    i = r._("gallery_from_domain", u);
                                    break;
                                case "link":
                                    i = r._("gallery_links_from_domain", u)
                            }
                            var c = void 0;
                            if (n.extensions) {
                                var l = Object.keys(n.extensions);
                                l.sort((function (e, t) {
                                    return n.extensions[e] - n.extensions[t]
                                }));
                                var d = [];
                                l.forEach((function (e) {
                                    var t = r._("number_type", ["" + n.extensions[e], e.toUpperCase()]);
                                    d.push(t)
                                })), c = r._("gallery_files_types", d.length > 0 && d.join(", ") || "" + n.urls.length)
                            }
                            var f = "gallery:" + a.hashHex(e.pageUrl) + ":" + t;
                            o.dispatch("hit.new", Object.assign({}, n, {
                                id: f,
                                topUrl: e.pageUrl,
                                title: i,
                                description: c,
                                mouseTrack: !0
                            })), s.getProxyHeaders(e.pageUrl).then((function (e) {
                                o.dispatch("hit.update", {
                                    id: f,
                                    changes: e
                                })
                            }))
                        }))
                    },
                    galleryHighlight: async function (e) {
                        i.scripting.insertCSS({
                            target: await u(),
                            css: l(e)
                        })
                    },
                    galleryUnhighlight: async function (e) {
                        i.scripting.removeCSS({
                            target: await u(),
                            css: l(e)
                        })
                    }
                }), i.tabs.onUpdated.addListener((function (e, t, n) {
                    "complete" === t.status && r.prefs.medialinkAutoDetect && c(e)
                }))
            },
            50: (e, t, n) => {
                "use strict";

                function r(e, t, n) {
                    return t in e ? Object.defineProperty(e, t, {
                        value: n,
                        enumerable: !0,
                        configurable: !0,
                        writable: !0
                    }) : e[t] = n, e
                }
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.reducer = function () {
                    var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                        t = arguments[1];

                    function n(e) {
                        var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {},
                            n = e;
                        !e.referrer && t.pageUrl && (t = Object.assign({}, t, {
                            referrer: t.pageUrl
                        })), Object.keys(t).every((function (n) {
                            return s.equals(e[n], t[n])
                        })) || (e = Object.assign({}, e, t));
                        var r = d(e);
                        if ("running" != e.status && r != e.status) {
                            var o = e.status;
                            if (e === n ? e = Object.assign({}, e, {
                                    status: r
                                }) : e.status = r, "orphan" == r && "orphan" != o) {
                                var c = Date.now(),
                                    l = 1e3 * i.prefs.orphanExpiration;
                                e.orphanT0 = c, e.orphanT = c + l, setTimeout((function () {
                                    u.dispatch("hit.orphanTimeout", e.id)
                                }), l + 100)
                            }
                        }
                        var f = a.availableActions(e);
                        return s.equals(f, e.actions) || (e === n ? e = Object.assign({}, e, {
                            actions: f
                        }) : e.actions = f), e
                    }

                    function o(t, i) {
                        Array.isArray(t) || (t = [t]), t.forEach((function (t) {
                            var o = e[t];
                            if (o) {
                                var a = e,
                                    s = n(o, i);
                                e === a && (e = Object.assign({}, e, r({}, t, s)))
                            }
                        }))
                    }
                    switch (t.type) {
                        case "hit.new":
                            var c = l.checkHitBlacklisted(t.payload),
                                f = "number" == typeof t.payload.length && t.payload.length < i.prefs.mediaweightMinSize;
                            if (!c && !f) {
                                var p = t.payload.id,
                                    h = e[p],
                                    g = n(h || {
                                        status: "active"
                                    }, t.payload);
                                g !== h && (e = Object.assign({}, e, r({}, p, g)))
                            }
                            break;
                        case "hits.urlUpdated":
                            var m = e;
                            Object.keys(e).forEach((function (t) {
                                var r = e[t],
                                    i = n(r);
                                i !== r && (m === e && (e = Object.assign({}, e)), e[t] = i)
                            }));
                            break;
                        case "hit.update":
                            var v = t.payload;
                            o(v.id, v.changes);
                            break;
                        case "hit.updateRunning":
                            var y = t.payload,
                                b = y.id,
                                w = y.runningDelta,
                                k = e[b];
                            if (k) {
                                var x = k.running || 0,
                                    A = {
                                        running: (x || 0) + w
                                    };
                                0 == x && (A.status = "running"), 0 === A.running && (A.status = "active"), (e = Object.assign({}, e))[b] = n(k, A)
                            }
                            break;
                        case "hit.deleteProps":
                            var _ = t.payload,
                                O = _.id,
                                I = _.propsToBeDeleted;
                            Array.isArray(O) || (O = [O]), Array.isArray(I) || (I = [I]);
                            var P = e;
                            O.forEach((function (t) {
                                var n = e[t];
                                if (n) {
                                    var r = n;
                                    I.forEach((function (t) {
                                        void 0 !== r[t] && (r === n && (r = Object.assign({}, n), e === P && (e = Object.assign({}, e)), e[r.id] = r), delete r[t])
                                    }))
                                }
                            }));
                            break;
                        case "hit.updateOriginal":
                            var C = t.payload,
                                S = C.id,
                                j = C.changes,
                                E = [];
                            Object.keys(e).forEach((function (t) {
                                var n = e[t];
                                t !== S && S !== n.originalId || E.push(t)
                            })), o(E, j);
                            break;
                        case "hit.delete":
                            var D = e,
                                T = t.payload;
                            Array.isArray(T) || (T = [T]), T.forEach((function (t) {
                                e[t] && (e === D && (e = Object.assign({}, e)), delete e[t])
                            }));
                            break;
                        case "hit.orphanTimeout":
                            var R = t.payload,
                                q = e[R];
                            q && "orphan" == q.status && !isNaN(q.orphanT) && Date.now() > q.orphanT && delete(e = Object.assign({}, e))[R]
                    }
                    return e
                }, t.progressReducer = function () {
                    var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                        t = arguments[1];
                    switch (t.type) {
                        case "hit.progress":
                            e[t.payload.id] !== t.payload.progress && (e = Object.assign({}, e, r({}, t.payload.id, t.payload.progress)));
                            break;
                        case "hit.clear-progress":
                            void 0 !== e[t.payload] && delete(e = Object.assign({}, e))[t.payload]
                    }
                    return e
                }, t.statusFromUrl = d, t.create = function (e) {
                    u.dispatch("hit.new", e)
                }, t.update = f, t.updateRunning = function (e, t) {
                    u.dispatch("hit.updateRunning", {
                        id: e,
                        runningDelta: t
                    })
                }, t.deleteProps = function (e, t) {
                    u.dispatch("hit.deleteProps", {
                        id: e,
                        propsToBeDeleted: t
                    })
                }, t.updateOriginal = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                    u.dispatch("hit.updateOriginal", {
                        id: e,
                        changes: t
                    })
                }, t.updateProgress = function (e, t) {
                    null === t ? u.dispatch("hit.clear-progress", e) : u.dispatch("hit.progress", {
                        id: e,
                        progress: t
                    })
                }, t.setHitOperation = function (e, t) {
                    var n = u.getHit(e);
                    n && n.operation !== t && f(e, {
                        operation: t
                    })
                };
                var i = n(3),
                    o = n(1),
                    a = n(189),
                    s = n(7),
                    u = n(12),
                    c = n(49),
                    l = n(224);

                function d(e) {
                    var t = e.status,
                        n = c.current(),
                        r = n.url,
                        i = n.urls;
                    return "active" == e.status && e.topUrl != r ? t = e.topUrl in i ? "inactive" : "orphan" : "inactive" != e.status || e.topUrl in i ? "inactive" != e.status && "orphan" != e.status || e.topUrl != r || (t = "active") : t = "orphan", "orphan" == t && e.pinned && (t = "pinned"), t
                }

                function f(e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                    u.dispatch("hit.update", {
                        id: e,
                        changes: t
                    })
                }
                o.listen({
                    actionCommand: function (e, t) {
                        var n = u.getHit(t);
                        return a.execute(e, n)
                    },
                    clearHits: function (e) {
                        var t = [],
                            n = u.getHits();
                        for (var r in n) {
                            var i = n[r];
                            ("all" == e && "running" != i.status && "pinned" != i.status || "pinned" == e && "pinned" == i.status || "inactive" == e && "inactive" == i.status || "orphans" == e && "orphan" == i.status) && t.push(r)
                        }
                        u.dispatch("hit.delete", t)
                    }
                })
            },
            218: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
                        return typeof e
                    } : function (e) {
                        return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                    },
                    i = function () {
                        function e(e, t) {
                            for (var n = 0; n < t.length; n++) {
                                var r = t[n];
                                r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                            }
                        }
                        return function (t, n, r) {
                            return n && e(t.prototype, n), r && e(t, r), t
                        }
                    }();
                t.canStop = function (e) {
                    var t = g[e];
                    if (!t) return !1;
                    return t.recording
                }, t.stopRecording = function (e) {
                    var t = g[e];
                    if (!t) return;
                    t.endRecording()
                }, t.handleMaster = function (e, t) {
                    e.walkThrough((function (e, n) {
                        var r = u.hashHex(e),
                            i = "hls:" + r,
                            o = g[i];
                        o || (o = m(t, r, e, n), g[i] = o)
                    }))
                }, t.handleMedia = v, t.getChunkSet = function (e) {
                    return g["hls:" + u.hashHex(e.url)] || null
                };
                var o = n(3),
                    a = n(12),
                    s = n(50),
                    u = n(7),
                    c = n(78),
                    l = n(239),
                    d = n(221),
                    f = n(32),
                    p = n(34),
                    h = new RegExp("^https://soundcloud.com/"),
                    g = {};
                setInterval((function () {
                    var e = [];
                    for (var t in g) {
                        a.getHit(t) ? e.push(g[t].chunks.length) : delete g[t]
                    }
                }), 6e4);

                function m(e, t, n, r) {
                    var i = "hls:" + t,
                        s = "mp4",
                        u = !1;
                    e.topUrl && h.test(e.topUrl) ? (u = !0, s = "mp3") : o.prefs.hlsDownloadAsM2ts && (s = "m2ts");
                    var c = Object.assign({}, e, {
                        id: i,
                        extension: s,
                        hls: r,
                        url: n,
                        length: null,
                        chunked: "hls",
                        durationFloat: 0,
                        mp3Direct: u
                    });
                    c.mediaManifest = n;
                    var l = r["EXT-X-MEDIA"];
                    l && (c.quality = l.NAME || c.quality);
                    var d = r["EXT-X-STREAM-INF"];
                    return d && (c.size = d.RESOLUTION || c.size, c.bitrate = parseInt(d.BANDWIDTH) || c.bitrate), a.dispatch("hit.new", c), new y(c)
                }

                function v(e, t, n) {
                    var r = u.hashHex(n),
                        i = g["hls:" + r];
                    i || (i = m(t, r, n, {}), g["hls:" + r] = i);
                    var c = a.getHit("hls:" + r);
                    if (c) {
                        var l = 0;
                        i.chunkDuration = 1e3, i.recording || o.prefs.hlsRememberPrevLiveChunks || (i.chunks = [], i.chunksMap = {}, i.segmentsCount = 0), e.walkThrough((function (t, n) {
                            var r = n.EXTINF;
                            if (r) {
                                var o = Math.round(1e3 * parseFloat(r));
                                o > i.chunkDuration && (i.chunkDuration = o)
                            }
                            var a = u.hashHex(t);
                            if (!(a in i.chunksMap)) {
                                l++, i.chunksMap[a] = 1;
                                var s = {
                                        url: t,
                                        index: i.chunks.length
                                    },
                                    d = n["EXT-X-KEY"];
                                d && "NONE" != d.METHOD && (s.encrypt = d, s.iv = parseInt(e.tags["EXT-X-MEDIA-SEQUENCE"] || "0") + s.index), i.chunks.push(s), r && (c.durationFloat += parseFloat(r), c.duration = Math.round(c.durationFloat))
                            }
                        })), s.update(c.id, {
                            durationFloat: c.durationFloat,
                            duration: c.duration
                        }), l > 0 && (i.segmentsCount += l, i.handle())
                    }
                }
                var y = function (e) {
                    function t(e) {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, t);
                        var n = function (e, t) {
                                if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                                return !t || "object" != typeof t && "function" != typeof t ? e : t
                            }(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e)),
                            r = u.hashHex(e.url);
                        return n.id = "hls:" + r, n.hit = Object.assign({}, e, {
                            chunked: "hls",
                            descrPrefix: o._("hls_streaming")
                        }), n.chunksMap = {}, n.chunks = [], n.segmentsCount = 0, n.doNotReportDownloadChunkErrors = !0, n.chunkDuration = 1e3, a.dispatch("hit.new", n.hit), n
                    }
                    return function (e, t) {
                        if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                        e.prototype = Object.create(t && t.prototype, {
                            constructor: {
                                value: e,
                                enumerable: !1,
                                writable: !0,
                                configurable: !0
                            }
                        }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                    }(t, e), i(t, [{
                        key: "download",
                        value: function (e, t, n, r, i) {
                            var a = this;
                            this.aborted = !1, this.action = e, this.specs = t, this.successFn = function () {
                                n.apply(a, arguments)
                            }, this.errorFn = function () {
                                r.apply(a, arguments)
                            }, this.progressFn = i, this.downloadTarget = t[0].fileName, this.nextTrackId = 1, this.processedSegmentsCount = 0, this.recording = !0, this.segmentsCount || this.requestMediaManifest(), this.hit.mp3Direct || o.prefs.hlsDownloadAsM2ts ? (a.recording = !0, a.handle()) : c.writeFileHeader(this, (function (e) {
                                e ? r(e) : (a.recording = !0, a.handle())
                            })), e.hit.abortFn = function () {
                                a.actionAbortFn(a.downloadTarget + ".part")
                            }
                        }
                    }, {
                        key: "downloadChunk",
                        value: function (e, t) {
                            e.encrypt ? this.downloadEncryptedChunk(e, t) : l.MP2TChunkset.prototype.downloadChunk.call(this, e, t)
                        }
                    }, {
                        key: "downloadEncryptedChunk",
                        value: function (e, t) {
                            var n = this;

                            function i(r, i, o) {
                                if (r) return t.call(n, r, e);
                                l.MP2TChunkset.prototype.downloadChunk.call(n, e, (function (r) {
                                    if (r) return t.call(n, r, e);
                                    crypto.subtle.decrypt({
                                        name: "AES-CBC",
                                        iv: o
                                    }, i, e.data).then((function (r) {
                                        e.data = new Uint8Array(r), t.call(n, null, e)
                                    })).catch((function (r) {
                                        t.call(n, r, e)
                                    }))
                                }))
                            }
                            if ("AES-128" != e.encrypt.METHOD) return t.call(this, new Error("HLS encryption method " + e.encrypt.METHOD + " is not supported"), e);
                            if (!e.encrypt.URI) return t.call(this, new Error("HLS encryption missing key URI"), e);
                            var a = parseInt(e.encrypt.IV || e.iv),
                                s = new Uint8Array(16);
                            f.WriteInt32(s, 12, a), this.keys || (this.keys = {}), Array.isArray(this.keys[e.encrypt.URI]) ? this.keys[e.encrypt.URI].push((function (e, t) {
                                i.call(n, e, t, s)
                            })) : "object" == r(this.keys[e.encrypt.URI]) ? i(null, this.keys[e.encrypt.URI], s) : (this.keys[e.encrypt.URI] = [function (e, t) {
                                i.call(n, e, t, s)
                            }], o.prefs.chunkedCoappDataRequests ? p.requestBinary(e.encrypt.URI, {
                                headers: n.hit.headers || [],
                                proxy: o.prefs.coappUseProxy && n.hit.proxy || null
                            }).then((function (r) {
                                crypto.subtle.importKey("raw", r, "aes-cbc", !0, ["decrypt"]).then((function (t) {
                                    var r = n.keys[e.encrypt.URI];
                                    n.keys[e.encrypt.URI] = t, r.forEach((function (e) {
                                        e(null, t)
                                    }))
                                })).catch((function (r) {
                                    t.call(n, r, e)
                                }))
                            })).catch((function (r) {
                                t.call(n, r, e)
                            })) : u.downloadToByteArray(e.encrypt.URI, {}, !0).then((function (r) {
                                crypto.subtle.importKey("raw", r, "aes-cbc", !0, ["decrypt"]).then((function (t) {
                                    var r = n.keys[e.encrypt.URI];
                                    n.keys[e.encrypt.URI] = t, r.forEach((function (e) {
                                        e(null, t)
                                    }))
                                })).catch((function (r) {
                                    t.call(n, r, e)
                                }))
                            })).catch((function (r) {
                                t.call(n, r, e)
                            })))
                        }
                    }, {
                        key: "outOfChunks",
                        value: function () {
                            var e, t = this,
                                n = Math.max(2 * t.chunkDuration, 1e3 * o.prefs.hlsEndTimeout);
                            t.requestMediaManifest(), setTimeout((function () {
                                t.requestMediaManifest()
                            }), n / 2), this.endTimer && clearTimeout(this.endTimer), e = this.segmentsCount, t.endTimer = setTimeout((function () {
                                t.recording && e == t.segmentsCount && l.MP2TChunkset.prototype.outOfChunks.call(t)
                            }), n)
                        }
                    }, {
                        key: "requestMediaManifest",
                        value: async function () {
                            if (this.hit && this.recording)
                                if (o.prefs.chunkedCoappManifestsRequests || p.isProbablyAvailable()) try {
                                    var e = await p.request(this.hit.url, {
                                            headers: this.hit.headers || [],
                                            proxy: o.prefs.coappUseProxy && this.hit.proxy || null
                                        }),
                                        t = d.get(e, this.hit.url);
                                    t && t.isMedia() && v(t, this.hit, this.hit.url)
                                } catch (e) {
                                    console.warn("media manifest request for", this.hit.url, "failed:", e.message), this.endRecording()
                                } else {
                                    var n = await u.request({
                                        headers: this.his.headers,
                                        url: this.hit.url,
                                        isPrivate: this.hit.isPrivate
                                    });
                                    if (n.ok && this.hit.url) {
                                        var r = await n.text(),
                                            i = d.get(r, this.hit.url);
                                        i && i.isMedia() && v(i, this.hit, this.hit.url)
                                    }
                                }
                        }
                    }, {
                        key: "endRecording",
                        value: function (e) {
                            this.recording && (this.recording = !1, this.finalize(e || null, (function (t) {
                                t && console.warn("Uncaught endRecording error:", t, e)
                            })))
                        }
                    }, {
                        key: "finalize",
                        value: function (e, t) {
                            l.MP2TChunkset.prototype.finalize.call(this, e, t), delete g[this.id]
                        }
                    }, {
                        key: "mediaTimeoutTriggered",
                        value: function () {
                            this.endRecording()
                        }
                    }, {
                        key: "setNewId",
                        value: function () {
                            delete g[this.id], l.MP2TChunkset.prototype.setNewId.call(this), g[this.id] = this, this.requestMediaManifest()
                        }
                    }]), t
                }(l.MP2TChunkset)
            },
            190: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.reducer = function () {
                    var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : [],
                        t = arguments[1];
                    switch (t.type) {
                        case "log.new":
                            e = e.concat([Object.assign({
                                key: ++a
                            }, t.payload)]);
                            break;
                        case "log.clear":
                            e = []
                    }
                    return e
                }, t.error = function (e) {
                    s(e, "error")
                }, t.log = s, t.clear = u, t.logDetails = c, t.getEntry = l;
                var r = n(3),
                    i = r.browser,
                    o = n(12),
                    a = 0;

                function s(e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : "log";
                    if (e instanceof Error) {
                        var n = "";
                        e.fileName && e.lineNumber && ((n = e.fileName + ":" + e.lineNumber).columnNumber && (n += ":" + e.columnNumber), n += "\n"), e.stack && (n += e.stack), e = {
                            message: e.message,
                            details: e.details || n || void 0,
                            videoTitle: e.videoTitle || void 0
                        }
                    } else e = "string" == typeof e ? {
                        message: e
                    } : {
                        message: e.message || "" + e,
                        details: e.details || void 0
                    };
                    o.dispatch("log.new", Object.assign(e, {
                        type: t
                    }))
                }

                function u() {
                    o.dispatch("log.clear")
                }

                function c(e) {
                    r.rpc.call("main", "embed", i.runtime.getURL("content/logdetails-embed.html?panel=logdetails#" + encodeURIComponent(e)))
                }

                function l(e) {
                    var t = null;
                    if (o.getLogs().forEach((function (n) {
                            n.key == e && (t = n)
                        })), t) return t;
                    throw new Error("Log entry not found")
                }
                r.rpc.listen({
                    clearLogs: u,
                    logDetails: c,
                    getLogEntry: l
                })
            },
            221: (e, t) => {
                "use strict";
                var n = new RegExp("^#(EXT[^\\s:]+)(?::(.*))"),
                    r = new RegExp('^\\s*([^=\\s]+)\\s*=\\s*(?:"([^"]*?)"|([^,]*)),?s*(.*?)s*$'),
                    i = new RegExp('^\\s*"(.*)"\\s*$');

                function o() {}

                function a() {}
                o.prototype = {
                    init: function () {
                        this.tags = {}, this.segments = [], this.valid = !1
                    },
                    parse: function (e, t) {
                        var r = e.split(/[\r\n]+/);
                        if (0 != r.length && "#EXTM3U" == r[0].trim()) {
                            this.master = !0;
                            for (var i = [], o = {}, a = 1; a < r.length; a++) {
                                var s = r[a].trim();
                                if ("" != s)
                                    if ("#" == s[0]) {
                                        if (0 != s.indexOf("#EXT")) continue;
                                        var u = n.exec(s);
                                        if (!u) continue;
                                        "EXTINF" == u[1] && (this.master = !1), o[u[1]] = u[2]
                                    } else i.push({
                                        url: new URL(s, t).href,
                                        tags: Object.assign({}, o)
                                    })
                            }
                            if (0 != i.length) {
                                for (var c in i[0].tags) {
                                    for (var l = i[0].tags[c], d = !0, f = 1; f < i.length; f++) {
                                        if (i[f].tags[c] !== l) {
                                            d = !1;
                                            break
                                        }
                                    }
                                    d && (this.tags[c] = this.parseAttrs(l))
                                }
                                for (var p = 0; p < i.length; p++) {
                                    var h = i[p],
                                        g = {
                                            url: h.url,
                                            tags: {}
                                        };
                                    for (var m in h.tags) void 0 === this.tags[m] && (g.tags[m] = this.parseAttrs(h.tags[m]));
                                    this.segments.push(g)
                                }
                                this.valid = !0
                            }
                        }
                    },
                    parseAttrs: function (e) {
                        var t = i.exec(e);
                        if (t) return t[1];
                        if (e.indexOf("=") < 0) return e;
                        for (var n = {}, o = e; o.length > 0;) {
                            var a = r.exec(o);
                            if (!a) break;
                            var s = a[1],
                                u = a[2] || a[3];
                            n[s] = u, o = a[4]
                        }
                        return n
                    },
                    isMaster: function () {
                        return this.valid && this.master
                    },
                    isMedia: function () {
                        return this.valid && !this.master
                    },
                    walkThrough: function (e) {
                        var t = this;
                        this.segments.forEach((function (n, r) {
                            e(n.url, Object.assign({}, t.tags, n.tags), r)
                        }))
                    }
                }, a.prototype = new o, a.prototype.parse = function (e, t) {
                    try {
                        var n = 0 == t.indexOf("https"),
                            r = JSON.parse(e);
                        r.hls_url && !n && this.segments.push({
                            url: r.hls_url,
                            tags: {}
                        }), r.https_hls_url && n && this.segments.push({
                            url: r.https_hls_url,
                            tags: {}
                        }), this.segments.length > 0 && (this.valid = !0, this.master = !0)
                    } catch (e) {
                        console.warn("PsJsonM3U8::parse failed", e)
                    }
                }, t.get = function (e, t) {
                    var n = new o;
                    return n.init(), n.parse(e, t), n.valid && n.tags && n.tags["EXT-X-KEY"] && n.tags["EXT-X-KEY"].URI && (n.tags["EXT-X-KEY"].URI = new URL(n.tags["EXT-X-KEY"].URI, t).href), n.valid && n || null
                }, t.getPsJson = function (e, t) {
                    var n = new a;
                    return n.init(), n.parse(e, t), n.valid && n || null
                }
            },
            244: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.updateHits = w, t.ExecuteDefault = k;
                var r = n(64),
                    i = n(8),
                    o = n(3),
                    a = o.browser,
                    s = n(189),
                    u = n(12),
                    c = n(7),
                    l = n(225),
                    d = n(79),
                    f = n(49),
                    p = new RegExp("(\\d+)x(\\d+)"),
                    h = a.menus || a.contextMenus;

                function g(e, t) {
                    if (e.bitrate && t.bitrate && e.bitrate != t.bitrate) return t.bitrate - e.bitrate;
                    var n = p.exec(e.size);
                    if (n) {
                        var r = p.exec(t.size);
                        if (r && (n[1] != r[1] || n[2] != r[2])) return parseInt(r[1]) * parseInt(r[2]) - parseInt(n[1]) * parseInt(n[2])
                    }
                    return 0
                }

                function m(e) {
                    if (e.description) return e.description;
                    var t, n, r, i, a = [];
                    if (e.descrPrefix && a.push(e.descrPrefix), e.adp && a.push("ADP"), e.size && a.push(e.size), e.duration && a.push((t = e.duration, n = Math.floor(t / 3600), r = Math.floor(t % 3600 / 60), i = t % 60, n > 0 ? n + ":" + ("00" + r).substr(-2) + ":" + ("00" + i).substr(-2) : r + ":" + ("00" + i).substr(-2))), e.quality) {
                        var s = o._("quality_" + e.quality);
                        "" == s && (s = e.quality.toUpperCase()), a.push(s)
                    }
                    if (e.bitrate) {
                        var u = e.bitrate,
                            c = "bps";
                        e.bitrate > 1e7 ? (c = "Mbps", u = Math.round(e.bitrate / 1e6)) : e.bitrate > 1e6 ? (c = "Mbps", u = Math.round(e.bitrate / 1e5) / 10) : e.bitrate > 1e4 ? (c = "Kbps", u = Math.round(e.bitrate / 1e3)) : e.bitrate > 1e3 && (c = "Kbps", u = Math.round(e.bitrate / 100) / 10), a.push(u + c)
                    }
                    var l = function (e) {
                        return e.length ? e.length > 1048576 ? o._("MB", [Math.round(10 * e.length / 1048576) / 10]) : e.length > 1024 ? o._("KB", [Math.round(10 * e.length / 1024) / 10]) : o._("Bytes", [e.length]) : null
                    }(e);
                    return l && a.push(l), e.mediaDomain && a.push(o._("from_domain", [e.mediaDomain])), "audio" == e.type && a.push(o._("audio_only")), e.extension && (e.originalExt && e.originalExt != e.extension && a.push(e.originalExt.toUpperCase() + ">" + e.extension.toUpperCase()), a.push(e.extension.toUpperCase())), a.join(" - ")
                }
                var v = [],
                    y = null,
                    b = c.Concurrent();

                function w(e) {
                    y = null;
                    var t = Object.keys(e).map((function (t) {
                            return e[t]
                        })).filter((function (e) {
                            return "active" == e.status
                        })),
                        n = {};
                    t.forEach((function (e) {
                        var t = e.group || e.id;
                        void 0 === n[t] && (n[t] = []), n[t].push(e)
                    }));
                    var u = [],
                        c = s.describeAll();

                    function l(e) {
                        return new Promise((function (t, n) {
                            h.create(e, (function () {
                                t()
                            }))
                        }))
                    }
                    Object.keys(n).forEach((function (e) {
                        var t = n[e];
                        t[0].urls ? u.push({
                            id: e,
                            title: t[0].title
                        }) : (t.sort(g), u.push({
                            id: "group-" + e,
                            title: t[0].title,
                            enabled: !1
                        }), t.forEach((function (e) {
                            y || (y = e.id), u.push({
                                id: e.id,
                                title: m(e),
                                icons: {
                                    18: "content/" + c[e.actions[0]].icon18
                                }
                            })
                        })))
                    })), 0 == u.length && u.push({
                        id: "vdh-no-media",
                        title: o._("no_media_current_tab"),
                        enabled: !1
                    }), u.push({
                        id: "separator",
                        type: "separator"
                    }), u.push({
                        id: "vdh-smartnaming",
                        title: o._("smartnaming_rule")
                    }), u.push({
                        id: "separator2",
                        type: "separator"
                    }), u.push({
                        id: "vdh-settings",
                        title: o._("settings"),
                        icons: {
                            64: "content/images/icon-settings-64.png"
                        }
                    }), u.push({
                        id: "vdh-about",
                        title: o._("about"),
                        icons: {
                            64: "content/images/icon-about-64.png"
                        }
                    }), "chrome" != i.buildOptions.browser && u.push({
                        id: "vdh-sites",
                        title: o._("supported_sites"),
                        icons: {
                            64: "content/images/icon-sites-list-64.png"
                        }
                    }), u.push({
                        id: "vdh-convert",
                        title: o._("convert_local_files"),
                        icons: {
                            64: "content/images/icon-action-convert-b-64.png"
                        }
                    }), u.push({
                        id: "vdh-merge",
                        title: o._("merge_local_files"),
                        icons: {
                            64: "content/images/icon-merger-64.png"
                        }
                    }), u.push({
                        id: "vdh-analyze",
                        title: o._("analyze_page"),
                        icons: {
                            64: "content/images/icon-photo-64.png"
                        }
                    }), o.prefs.contextMenuEnabled || o.isBrowser("firefox") && o.prefs.toolsMenuEnabled ? r(u, v) || (v = u, b((function () {
                        return h.removeAll().then((function () {
                            var e = [];
                            return o.prefs.contextMenuEnabled && e.push(l({
                                id: "vdh-main",
                                title: o._("title"),
                                contexts: ["all"]
                            }).then((function () {
                                return Promise.all(u.map((function (e) {
                                    var t = Object.assign({
                                        parentId: "vdh-main"
                                    }, e);
                                    return a.menus || (delete t.icons, (t.contexts || []).indexOf("selection") < 0 && (t.contexts = (t.contexts || []).concat(["all"]))), l(t)
                                })))
                            }))), o.isBrowser("firefox") && o.prefs.toolsMenuEnabled && e.push(l({
                                id: "vdh-main-tools",
                                title: o._("title"),
                                contexts: ["tools_menu"]
                            }).then((function () {
                                return Promise.all(u.map((function (e) {
                                    return l(Object.assign({
                                        parentId: "vdh-main-tools"
                                    }, e, {
                                        id: e.id + "-tools"
                                    }))
                                })))
                            }))), Promise.all(e)
                        }))
                    }))) : v.length && (h.removeAll(), v = [])
                }

                function k() {
                    if (y) {
                        var e = u.getHit(y);
                        e && s.execute(e.actions[0], e)
                    }
                }
                w({}), o.prefs.on("contextMenuEnabled", (function () {
                    w(u.getHits())
                })), o.prefs.on("toolsMenuEnabled", (function () {
                    w(u.getHits())
                })), h.onClicked.addListener((function (e, t) {
                    switch (e.menuItemId) {
                        case "vdh-settings":
                        case "vdh-settings-tools":
                            o.ui.open("settings", {
                                type: "tab",
                                url: "content/settings.html"
                            });
                            break;
                        case "vdh-about":
                        case "vdh-about-tools":
                            o.ui.open("about", {
                                type: "panel",
                                url: "content/about.html"
                            });
                            break;
                        case "vdh-sites":
                        case "vdh-sites-tools":
                            f.gotoOrOpenTab("https://www.downloadhelper.net/sites");
                            break;
                        case "vdh-convert":
                        case "vdh-convert-tools":
                            s.convertLocal();
                            break;
                        case "vdh-merge":
                            s.mergeLocal();
                            break;
                        case "vdh-analyze":
                        case "vdh-analyze-tools":
                            l.analyzePage();
                            break;
                        case "vdh-smartnaming":
                        case "vdh-smartnaming-tools":
                            d.defineInPage();
                            break;
                        default:
                            var n = /^(.*)\-tools$/.exec(e.menuItemId),
                                r = u.getHit(n && n[1] || e.menuItemId);
                            r && s.execute(r.actions[0], r)
                    }
                })), a.commands.onCommand.addListener((function (e) {
                    "default-action" == e && k()
                }))
            },
            239: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = function () {
                    function e(e, t) {
                        for (var n = 0; n < t.length; n++) {
                            var r = t[n];
                            r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                        }
                    }
                    return function (t, n, r) {
                        return n && e(t.prototype, n), r && e(t, r), t
                    }
                }();
                var i = n(3),
                    o = n(78),
                    a = n(50),
                    s = n(220),
                    u = s.Chunkset,
                    c = s.Codecs,
                    l = n(240).S("mp2t-worker.js", 6e4);
                t.MP2TChunkset = function (e) {
                    function t(e) {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, t);
                        var n = function (e, t) {
                            if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                            return !t || "object" != typeof t && "function" != typeof t ? e : t
                        }(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                        return n.rawAppendData = i.prefs.hlsDownloadAsM2ts || e.mp3Direct, n.codecs = c, n.captureRaw = i.prefs.mpegtsSaveRaw, n.captureRawStreams = i.prefs.mpegtsSaveRawStreams, n.endsOnSeenChunk = i.prefs.mpegtsEndsOnSeenChunk, n.pesTable = {}, n.processQueue = [], n.workerWorking = !1, n
                    }
                    return function (e, t) {
                        if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                        e.prototype = Object.create(t && t.prototype, {
                            constructor: {
                                value: e,
                                enumerable: !1,
                                writable: !0,
                                configurable: !0
                            }
                        }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                    }(t, e), r(t, [{
                        key: "toUint8ArrayArray",
                        value: function (e) {
                            var t = [];
                            return function e(n) {
                                if (Array.isArray(n))
                                    for (var r = 0; r < n.length; r++) e(n[r]);
                                else n.byteLength > 0 && t.push(new Uint8Array(n))
                            }(e), t
                        }
                    }, {
                        key: "processChunkData",
                        value: function (e, t) {
                            if (this.rawAppendData) return t(null, e);
                            var n = this;
                            this.processQueue.push({
                                    data: e,
                                    callback: t
                                }),
                                function e() {
                                    if (!n.workerWorking) {
                                        var t = n.processQueue.shift();
                                        if (t) {
                                            n.workerWorking = !0;
                                            var r = {
                                                processedChunksCount: n.processedChunksCount,
                                                codecs: n.codecs,
                                                pidTable: n.pidTable,
                                                pesTable: n.pesTable,
                                                pmtTable: n.pmtTable,
                                                dataOffset: n.dataOffset,
                                                nextTrackId: n.nextTrackId
                                            };
                                            l("processData", r, t.data).then((function (i) {
                                                Object.keys(r).forEach((function (e) {
                                                    n[e] = i.meta[e]
                                                })), i.data && t.callback(null, n.toUint8ArrayArray(i.data)), n.workerWorking = !1, e()
                                            }), (function (r) {
                                                t.callback(r), n.workerWorking = !1, e()
                                            }))
                                        }
                                    }
                                }()
                        }
                    }, {
                        key: "finalize",
                        value: function (e, t) {
                            this.aborted && (e = new Error("Aborted"));
                            var n = this;
                            if (e) u.prototype.finalize.call(this, e, t);
                            else if (this.rawAppendData) this.waitForWrittenData((function () {
                                o.finalize(n, null, n.downloadTarget, (function (e) {
                                    u.prototype.finalize.call(n, e, t)
                                }))
                            }));
                            else {
                                var r = 1 / 0,
                                    i = [];
                                if (this.walkThroughAvailPes((function (e) {
                                        i.push(e), e.tsMin < r && (r = e.tsMin)
                                    })), 0 == i.length) return void u.prototype.finalize.call(this, new Error("MP2T - No data received"), t);
                                i.forEach((function (e) {
                                    e.shiftTs = e.tsMin - r
                                })), this.action && this.action.hit && a.setHitOperation(this.action.hit.id, "finalizing"), this.waitForWrittenData((function () {
                                    o.finalize(n, i, n.downloadTarget, (function (e) {
                                        u.prototype.finalize.call(n, e, t)
                                    }))
                                }))
                            }
                        }
                    }, {
                        key: "getTrackDuration",
                        value: function (e) {
                            return e.durationSec ? Math.round(1e3 * e.durationSec) : e.declaredSampleRate ? Math.round(1e3 * e.sampleCount / (1024 * e.declaredSampleRate)) : e.sampleRate ? Math.round(1e3 * e.sampleCount / e.sampleRate) : Math.round(e.tsMax - e.tsMin)
                        }
                    }, {
                        key: "getTotalDuration",
                        value: function () {
                            var e = 0;
                            return this.walkThroughAvailPes((function (t) {
                                var n = this.getTrackDuration(t);
                                n > e && (e = n)
                            })), e
                        }
                    }, {
                        key: "walkThroughAvailPes",
                        value: function (e) {
                            for (var t in this.pesTable) {
                                var n = this.pesTable[t];
                                "started" == n.state && e.call(this, n)
                            }
                        }
                    }]), t
                }(u)
            },
            78: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
                    return typeof e
                } : function (e) {
                    return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                };
                t.finalize = function (e, n, c, l) {
                    function p() {
                        var t = [];
                        e.secDuration = 0, n.forEach((function (t) {
                            t.scaledDuration = 0, t.secDuration = 0, (t.stts || []).forEach((function (e) {
                                t.scaledDuration += e.c * e.d
                            })), e.timeScale && e.timeScale[t.trackId] && (t.timeScale = e.timeScale[t.trackId], t.secDuration = t.scaledDuration / t.timeScale), e.secDuration = Math.max(e.secDuration, t.secDuration)
                        })), n.forEach((function (n) {
                            var i = function (e, t) {
                                    if (e.init && e.init.tkhd && e.init.tkhd[t.trackId]) {
                                        var n = e.init.tkhd[t.trackId],
                                            r = Math.round(t.secDuration * t.timeScale);
                                        return o.WriteInt32(n, 20, r), a("tkhd", e.init.tkhd[t.trackId])
                                    }
                                    var i = new Uint8Array(84);
                                    o.WriteInt24(i, 1, 3), o.WriteInt32(i, 4, 0), o.WriteInt32(i, 8, 0), o.WriteInt32(i, 12, t.trackId), o.WriteInt32(i, 20, e.getTrackDuration(t)), o.WriteInt16(i, 32, 0), o.WriteInt16(i, 34, 0), "audio" == t.streamType && o.WriteInt16(i, 36, 256);
                                    o.WriteInt16(i, 38, 0), d(i, 40), "video" == t.streamType && (o.WriteInt16(i, 76, t.width), o.WriteInt16(i, 80, t.height));
                                    return a("tkhd", i)
                                }(e, n),
                                c = function (e, t) {
                                    var n = function (e, t) {
                                            if (e.init && e.init.mdhd && e.init.mdhd[t.trackId]) {
                                                var n = e.init.mdhd[t.trackId];
                                                return o.WriteInt32(n, 12, t.timeScale), o.WriteInt32(n, 16, Math.round(t.secDuration * t.timeScale)), a("mdhd", n)
                                            }
                                            var r = new Uint8Array(24);
                                            if (o.WriteInt32(r, 0, 0), o.WriteInt32(r, 4, 0), o.WriteInt32(r, 8, 0), "video" == t.streamType) {
                                                o.WriteInt32(r, 12, 9e4);
                                                var i = 90 * e.getTrackDuration(t);
                                                o.WriteInt32(r, 16, i)
                                            } else {
                                                o.WriteInt32(r, 12, t.declaredSampleRate);
                                                var s = 1024 * t.sampleCount;
                                                o.WriteInt32(r, 16, s)
                                            }
                                            return o.WriteInt16(r, 20, 21956), o.WriteInt16(r, 22, 0), a("mdhd", r)
                                        }(e, t),
                                        i = function (e, t) {
                                            if (e.init && e.init.hdlr && e.init.hdlr[t.trackId]) return a("hdlr", e.init.hdlr[t.trackId]);
                                            var n = "VideoHandler",
                                                r = new Uint8Array(25 + n.length);
                                            o.WriteInt32(r, 0, 0), "audio" == t.streamType && u("mhlr", r, 4);
                                            "audio" == t.streamType ? u("soun", r, 8) : "video" == t.streamType && u("vide", r, 8);
                                            return o.WriteInt32(r, 12, 0), o.WriteInt32(r, 16, 0), o.WriteInt32(r, 20, 0), u(n, r, 24), a("hdlr", r)
                                        }(e, t),
                                        c = function (e, t) {
                                            var n = "video" == t.streamType ? function (e, t) {
                                                    if (e.init && e.init.vmhd && e.init.vmhd[t.trackId]) return a("vmhd", e.init.vmhd[t.trackId]);
                                                    var n = new Uint8Array(12);
                                                    return o.WriteInt32(n, 0, 1), o.WriteInt16(n, 4, 0), o.WriteInt16(n, 6, 0), o.WriteInt16(n, 8, 0), o.WriteInt16(n, 10, 0), a("vmhd", n)
                                                }(e, t) : [],
                                                i = "audio" == t.streamType ? function (e, t) {
                                                    if (e.init && e.init.smhd && e.init.smhd[t.trackId]) return a("smhd", e.init.smhd[t.trackId]);
                                                    var n = new Uint8Array(8);
                                                    return o.WriteInt32(n, 0, 0), o.WriteInt16(n, 4, 0), o.WriteInt16(n, 6, 0), a("smhd", n)
                                                }(e, t) : [],
                                                u = function (e, t) {
                                                    if (e.init && e.init.dinf && e.init.dinf[t.trackId]) return a("dinf", e.init.dinf[t.trackId]);
                                                    var n = function (e, t) {
                                                        var n = new Uint8Array(8);
                                                        o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, 1);
                                                        var r = function (e, t) {
                                                            var n = new Uint8Array(4);
                                                            return o.WriteInt32(n, 0, 1), a("url ", n)
                                                        }();
                                                        return a("dref", [n, r])
                                                    }();
                                                    return a("dinf", n)
                                                }(e, t),
                                                c = function (e, t) {
                                                    var n = function (e, t) {
                                                            if (e.init && e.init.stsd && e.init.stsd[t.trackId]) return a("stsd", e.init.stsd[t.trackId]);
                                                            var n = new Uint8Array(8);
                                                            o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, 1);
                                                            var r = [];
                                                            "audio" == t.streamType ? "mp4a" == t.codec.strTag && (r = function (e, t) {
                                                                var n = new Uint8Array(28);
                                                                o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, 1), o.WriteInt32(n, 8, 0), o.WriteInt32(n, 12, 0), o.WriteInt16(n, 16, 2), o.WriteInt16(n, 18, 16), o.WriteInt16(n, 20, 0), o.WriteInt16(n, 22, 0);
                                                                var r = e.mpd && e.mpd.sample_rate || t.declaredSampleRate || t.sampleRate || 48e3;
                                                                o.WriteInt16(n, 24, r), o.WriteInt16(n, 26, 0);
                                                                var i = function (e, t) {
                                                                    var n = null;
                                                                    if (void 0 !== t.mp4aProfile && void 0 !== t.mp4aRateIndex && void 0 !== t.mp4aChannelCount) {
                                                                        var r = 0;
                                                                        r |= t.mp4aProfile << 11, r |= t.mp4aRateIndex << 7, r |= t.mp4aChannelCount << 3, n = new Uint8Array(2), o.WriteInt16(n, 0, r)
                                                                    }
                                                                    var i = new Uint8Array(36 + (n ? n.length + 5 : 0));
                                                                    o.WriteInt32(i, 0, 0), s(i, 4, 3, 27 + (n ? n.length + 5 : 0)), o.WriteInt16(i, 9, t.trackId), o.WriteInt8(i, 11, 0), s(i, 12, 4, 23), o.WriteInt8(i, 17, 64), o.WriteInt8(i, 18, 21), o.WriteInt24(i, 19, 0);
                                                                    var u = 0;
                                                                    if (t.sampleSizes)
                                                                        if (Array.isArray(t.sampleSizes))
                                                                            for (var c = 0; c < t.sampleSizes.length; c++) u += t.sampleSizes[c];
                                                                        else
                                                                            for (var l = 0; l < t.sampleSizes.length; l++) u += o.ReadInt32(t.sampleSizes.data, 4 * l);
                                                                    else if (Array.isArray(t.dataSizes))
                                                                        for (var d = 0; d < t.dataSizes.length; d++) u += t.dataSizes[d];
                                                                    else
                                                                        for (var f = 12; f < t.dataSizes.length; f += 4) u += o.ReadInt32(t.dataSizes.data, f);
                                                                    var p = Math.round(8 * u / t.durationSec);
                                                                    o.WriteInt32(i, 22, t.maxBitrate && Math.round(t.maxBitrate) || 2 * p), o.WriteInt32(i, 26, p), n ? (s(i, 30, 5, n.length), i.set(n, 35), s(i, 35 + n.length, 6, 1), o.WriteInt8(i, 40 + n.length, 2)) : (s(i, 30, 6, 1), o.WriteInt8(i, 35, 2));
                                                                    return a("esds", i)
                                                                }(0, t);
                                                                return a("mp4a", [n, i])
                                                            }(e, t)) : "video" == t.streamType && "avc1" == t.codec.strTag && (r = function (e, t) {
                                                                var n = new Uint8Array(78);
                                                                o.WriteInt32(n, 0, 0), o.WriteInt16(n, 4, 0), o.WriteInt16(n, 6, 1), o.WriteInt16(n, 8, 0), o.WriteInt16(n, 10, 0), o.WriteInt32(n, 12, 0), o.WriteInt32(n, 16, 0), o.WriteInt32(n, 20, 0), o.WriteInt16(n, 24, t.width), o.WriteInt16(n, 26, t.height), o.WriteInt32(n, 28, 4718592), o.WriteInt32(n, 32, 4718592), o.WriteInt32(n, 36, 0), o.WriteInt16(n, 40, 1), o.WriteInt16(n, 74, 24), o.WriteInt16(n, 76, 65535);
                                                                var r = function (e, t) {
                                                                    if (!t.sps || !t.pps) return [];
                                                                    var n = new Uint8Array(t.sps),
                                                                        r = new Uint8Array(t.pps),
                                                                        i = new Uint8Array(11 + n.length + r.length);
                                                                    o.WriteInt8(i, 0, 1), o.WriteInt8(i, 1, n[1]), o.WriteInt8(i, 2, n[2]), o.WriteInt8(i, 3, n[3]), o.WriteInt8(i, 4, 255), o.WriteInt8(i, 5, 225), o.WriteInt16(i, 6, n.length), i.set(n, 8);
                                                                    var s = 8 + n.length;
                                                                    return o.WriteInt8(i, s, 1), o.WriteInt16(i, s + 1, r.length), i.set(r, s + 3), a("avcC", i)
                                                                }(0, t);
                                                                return a("avc1", [n, r])
                                                            }(0, t));
                                                            return a("stsd", [n, r])
                                                        }(e, t),
                                                        i = function (e, t) {
                                                            if (t.sampleSizes) {
                                                                var n = !0,
                                                                    r = void 0;
                                                                if (Array.isArray(t.sampleSizes)) {
                                                                    r = t.sampleSizes[0];
                                                                    for (var i = 1; i < t.sampleSizes.length; i++)
                                                                        if (t.sampleSizes[i] != r) {
                                                                            n = !1;
                                                                            break
                                                                        }
                                                                } else {
                                                                    r = o.ReadInt32(t.sampleSizes.data, 0);
                                                                    for (var s = 1; s < t.sampleSizes.length; s++)
                                                                        if (o.ReadInt32(t.sampleSizes.data, 4 * s) != r) {
                                                                            n = !1;
                                                                            break
                                                                        }
                                                                }
                                                                var u = void 0;
                                                                if (n) u = new Uint8Array(12), o.WriteInt32(u, 4, r);
                                                                else if (u = new Uint8Array(12 + 4 * t.sampleSizes.length), Array.isArray(t.sampleSizes))
                                                                    for (var c = 0; c < t.sampleSizes.length; c++) o.WriteInt32(u, 12 + 4 * c, t.sampleSizes[c]);
                                                                else u.set(t.sampleSizes.data.subarray(0, 4 * t.sampleSizes.length), 12);
                                                                return o.WriteInt32(u, 8, t.sampleSizes.length), a("stsz", u)
                                                            }
                                                            var l = new Uint8Array(4 * t.dataSizes.length + 12);
                                                            if (o.WriteInt32(l, 4, 0), o.WriteInt32(l, 8, t.dataSizes.length), Array.isArray(t.dataSizes))
                                                                for (var d = 0; d < t.dataSizes.length; d++) o.WriteInt32(l, 12 + 4 * d, t.dataSizes[d]);
                                                            else l.set(t.dataSizes.data.subarray(0, 4 * t.dataSizes.length), 12);
                                                            return a("stsz", l)
                                                        }(0, t),
                                                        u = function (e, t) {
                                                            if (t.stts) {
                                                                var n = new Uint8Array(8 + 8 * t.stts.length);
                                                                o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, t.stts.length);
                                                                for (var r = 0; r < t.stts.length; r++) {
                                                                    var i = t.stts[r];
                                                                    o.WriteInt32(n, 8 + 8 * r, i.c), o.WriteInt32(n, 8 + 8 * r + 4, i.d)
                                                                }
                                                                return a("stts", n)
                                                            }
                                                            for (var s = [], u = Array.isArray(t.dataTimestamps) ? t.dataTimestamps.length : t.dataTimestamps.length / 2, c = 0; c < u;) {
                                                                var l = 0,
                                                                    d = 1;
                                                                if (Array.isArray(t.dataTimestamps))
                                                                    for (; c + d < u && t.dataTimestamps[c + d] <= t.dataTimestamps[c]; d++);
                                                                else
                                                                    for (; c + d < u && o.ReadInt64(t.dataTimestamps.data, 8 * (c + d)) <= o.ReadInt64(t.dataTimestamps.data, 8 * c); d++);
                                                                c + d < u && (l = Array.isArray(t.dataTimestamps) ? (t.dataTimestamps[c + d] - t.dataTimestamps[c]) / d : (o.ReadInt64(t.dataTimestamps.data, 8 * (c + d)) - o.ReadInt64(t.dataTimestamps.data, 8 * c)) / d, t.declaredSampleRate && (l = Math.round(l * t.declaredSampleRate / 9e4))), c += d, !l || 0 != s.length && s[s.length - 1].duration == l ? s.length > 0 && (s[s.length - 1].count += d) : s.push({
                                                                    duration: l,
                                                                    count: d
                                                                })
                                                            }
                                                            var f = new Uint8Array(8 + 8 * s.length);
                                                            o.WriteInt32(f, 0, 0), o.WriteInt32(f, 4, s.length);
                                                            for (var p = 0; p < s.length; p++) o.WriteInt32(f, 8 + 8 * p, s[p].count), o.WriteInt32(f, 12 + 8 * p, s[p].duration);
                                                            return a("stts", f)
                                                        }(0, t),
                                                        c = "video" == t.streamType ? function (e, t) {
                                                            var n = new Uint8Array(8 + 4 * t.keyFrames.length);
                                                            if (o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, t.keyFrames.length), Array.isArray(t.keyFrames))
                                                                for (var r = 0; r < t.keyFrames.length; r++) o.WriteInt32(n, 8 + 4 * r, t.keyFrames[r]);
                                                            else n.set(t.keyFrames.data.subarray(0, 4 * t.keyFrames.length), 8);
                                                            return a("stss", n)
                                                        }(0, t) : [],
                                                        l = function (e, t) {
                                                            if (t.stsc) {
                                                                var n = void 0,
                                                                    i = void 0;
                                                                if ("object" == r(t.stsc[0])) {
                                                                    i = t.stsc.length, n = new Uint8Array(8 + 12 * i), o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, i);
                                                                    for (var s = 0; s < t.stsc.length; s++) {
                                                                        var u = t.stsc[s];
                                                                        o.WriteInt32(n, 8 + 12 * s, u.first_chunk + 1), o.WriteInt32(n, 8 + 12 * s + 4, u.samples_per_chunk), o.WriteInt32(n, 8 + 12 * s + 8, u.sample_description_index + 1)
                                                                    }
                                                                } else {
                                                                    i = t.stsc.length / 3, n = new Uint8Array(8 + 12 * i), o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, i);
                                                                    for (var c = 0; c < t.stsc.length; c++) o.WriteInt32(n, 8 + 12 * c, t.stsc[3 * c]), o.WriteInt32(n, 8 + 12 * c + 4, t.stsc[3 * c + 1]), o.WriteInt32(n, 8 + 12 * c + 8, t.stsc[3 * c + 2])
                                                                }
                                                                return a("stsc", n)
                                                            }
                                                            var l = new Uint8Array(20);
                                                            return o.WriteInt32(l, 0, 0), o.WriteInt32(l, 4, 1), o.WriteInt32(l, 8, 1), o.WriteInt32(l, 12, 1), o.WriteInt32(l, 16, 1), a("stsc", l)
                                                        }(0, t),
                                                        d = e.multiMdat ? function (e, t) {
                                                            var n = "object" == r(t.dataOffsets[0]) ? t.dataOffsets.length : t.dataOffsets.length / 2,
                                                                i = new Uint8Array(8 + 8 * n);
                                                            o.WriteInt32(i, 0, 0), o.WriteInt32(i, 4, n);
                                                            for (var s = 0; s < n; s++) {
                                                                var u = void 0,
                                                                    c = (u = "object" == r(t.dataOffsets[s]) ? t.dataOffsets[s] : {
                                                                        b: t.dataOffsets[2 * s],
                                                                        o: t.dataOffsets[2 * s + 1]
                                                                    }).o + e.mdatOffsets[u.b];
                                                                o.WriteInt32(i, 8 + 8 * s, Math.floor(c / 4294967296)), o.WriteInt32(i, 12 + 8 * s, 4294967295 & c)
                                                            }
                                                            return a("co64", i)
                                                        }(e, t) : function (e, t) {
                                                            var n = "object" == r(t.dataOffsets[0]) ? t.dataOffsets.length : t.dataOffsets.length / 2,
                                                                i = new Uint8Array(8 + 4 * n);
                                                            o.WriteInt32(i, 0, 0), o.WriteInt32(i, 4, n);
                                                            for (var s = 0; s < n; s++) {
                                                                var u = void 0,
                                                                    c = (u = "object" == r(t.dataOffsets[s]) ? t.dataOffsets[s] : {
                                                                        b: t.dataOffsets[2 * s],
                                                                        o: t.dataOffsets[2 * s + 1]
                                                                    }).o + e.mdatOffsets[u.b];
                                                                o.WriteInt32(i, 8 + 4 * s, c)
                                                            }
                                                            return a("stco", i)
                                                        }(e, t);
                                                    return a("stbl", [n, u, c, l, i, d])
                                                }(e, t);
                                            return a("minf", [n, i, u, c])
                                        }(e, t);
                                    return a("mdia", [n, i, c])
                                }(e, n),
                                l = function (e, t) {
                                    var n = e.init && e.init.edts && e.init.edts[t.trackId];
                                    if (n) {
                                        var r = h("elst", n);
                                        if (r.length > 0) {
                                            var i = o.ReadInt32(r[0].data, 12);
                                            o.WriteInt32(r[0].data, 8, Math.round(1e3 * t.secDuration) - i)
                                        }
                                        return a("edts", n)
                                    }
                                    if (!t.shiftTs) return [];
                                    var s = function (e, t) {
                                        var n = new Uint8Array(20);
                                        return o.WriteInt32(n, 0, 0), o.WriteInt32(n, 4, 1), o.WriteInt32(n, 8, e.getTrackDuration(t)), o.WriteInt32(n, 12, t.shiftTs), o.WriteInt16(n, 16, 1), o.WriteInt16(n, 18, 0), a("elst", n)
                                    }(e, t);
                                    return a("edts", s)
                                }(e, n),
                                f = a("trak", [i, l, c]);
                            t.push(f)
                        }));
                        var c = function (e) {
                                if (e.init && e.init.mvhd) {
                                    var t = e.init.mvhd;
                                    return o.WriteInt32(t, 12, 1e3), o.WriteInt32(t, 16, Math.round(1e3 * e.secDuration)), a("mvhd", e.init.mvhd)
                                }
                                var n = new Uint8Array(100);
                                return o.WriteInt8(n, 0, 0), o.WriteInt24(n, 1, 0), o.WriteInt32(n, 4, 0), o.WriteInt32(n, 8, 0), o.WriteInt32(n, 12, 1e3), o.WriteInt32(n, 16, e.getTotalDuration()), o.WriteInt32(n, 20, 65536), o.WriteInt16(n, 24, 256), d(n, 36), o.WriteInt32(n, 72, 0), o.WriteInt32(n, 76, 0), o.WriteInt32(n, 80, 0), o.WriteInt32(n, 84, 0), o.WriteInt32(n, 88, 0), o.WriteInt32(n, 92, 0), o.WriteInt32(n, 96, e.getNextTrackId()), a("mvhd", n)
                            }(e),
                            p = function (e) {
                                if (e.init && e.init.iods && e.init.iods) return a("iods", e.init.iods);
                                var t = new Uint8Array(16);
                                return o.WriteInt32(t, 0, 0), o.WriteInt8(t, 4, 16), o.WriteInt32(t, 5, 2155905031), o.WriteInt16(t, 9, 79), o.WriteInt16(t, 11, 65535), o.WriteInt16(t, 13, 65278), o.WriteInt8(t, 15, 255), a("iods", t)
                            }(e),
                            g = a("moov", [c, p, t]);
                        i.File.open(e.downloadTarget + ".part", {
                            write: !0,
                            append: !1
                        }).then((function (t) {
                            t.setPosition(0, i.File.POS_END).then((function () {
                                f(t, g, (function (n) {
                                    t.close().catch((function () {})).then((function () {
                                        return new Promise((function (e, t) {
                                            setTimeout(e, 250)
                                        }))
                                    })).then((function () {
                                        if (n) l(n);
                                        else {
                                            var t = e.downloadTarget;
                                            i.File.move(t + ".part", t).then((function () {
                                                return l(null)
                                            }), l)
                                        }
                                    }))
                                }))
                            }), (function (e) {
                                l(e)
                            }))
                        }), (function (e) {
                            l(e)
                        }))
                    }
                    e.rawAppendData ? l(null) : e.currentDataBlockSize > 0 ? t.updateMdatLength(e, e.mdatLengthOffset, e.currentDataBlockSize, (function (e) {
                        if (e) return l(e);
                        p()
                    })) : p()
                }, t.mdatBox = function () {
                    return a("mdat", [])
                };
                var i = n(219),
                    o = n(32);

                function a(e, t) {
                    var n = c(t),
                        r = new Uint8Array(4);
                    return o.WriteInt32(r, 0, 8 + n), [r, u(e = (e + "    ").substring(0, 4)), t]
                }

                function s(e, t, n, r) {
                    var i = 3;
                    for (o.WriteInt8(e, t++, n); i > 0; i--) o.WriteInt8(e, t++, (r >>> 7 * i | 128) >>> 0);
                    o.WriteInt8(e, t++, 127 & r)
                }

                function u(e, t, n) {
                    t = t || new Uint8Array(e.length), n = n || 0;
                    for (var r = 0, i = e.length; r < i; r++) t[n + r] = 255 & e.charCodeAt(r);
                    return t
                }

                function c(e) {
                    if (Array.isArray(e)) {
                        for (var t = 0, n = 0, r = e.length; n < r; n++) t += c(e[n]);
                        return t
                    }
                    return e.length
                }

                function l(e) {
                    if (Array.isArray(e)) {
                        var t = new Uint8Array(c(e)),
                            n = 0;
                        return function e(r) {
                            if (Array.isArray(r))
                                for (var i = 0, o = r.length; i < o; i++) e(r[i]);
                            else t.set(r, n), n += r.length
                        }(e), t
                    }
                    return e
                }

                function d(e, t) {
                    [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824].forEach((function (n, r) {
                        o.WriteInt32(e, t + 4 * r, n)
                    }))
                }

                function f(e, t, n) {
                    t = l(t), e.write(t).then((function () {
                        n(null)
                    }), (function (e) {
                        n(e)
                    }))
                }
                t.length = c, t.flatten = l, t.firstBuffer = function e(t) {
                    return Array.isArray(t) ? 0 == t.length ? null : e(t[0]) : t
                };

                function p(e, t) {
                    t = t || e.length;
                    for (var n = 0, r = [];;) {
                        if (0 == t) return r;
                        if (t < 8) return null;
                        var i = o.ReadInt32(e, n),
                            a = 8;
                        if (1 == i && (i = o.ReadInt64(e, n + 8), a = 16), i > t || i < 8) return null;
                        var s = String.fromCharCode(o.ReadInt8(e, n + 4), o.ReadInt8(e, n + 5), o.ReadInt8(e, n + 6), o.ReadInt8(e, n + 7));
                        r.push({
                            name: s,
                            offset: n,
                            length: i,
                            dataOffset: n + a,
                            dataLength: i - a,
                            data: e.subarray(n + a, n + i)
                        }), n += i, t -= i
                    }
                }

                function h(e, t, n) {
                    var r = [],
                        i = p(t, n);
                    if (!i) return r;
                    for (var o = 0, a = i.length; o < a; o++) {
                        var s = i[o];
                        s.name == e && r.push(s)
                    }
                    return r
                }
                t.writeMulti = f, t.writeFileHeader = function (e, t) {
                    var n = void 0;
                    if (e.init && e.init.ftyp) n = a("ftyp", e.init.ftyp);
                    else {
                        var r = new Uint8Array(4);
                        o.WriteInt32(r, 0, 512), n = a("ftyp", [u("isom"), r, u("isomiso2avc1mp41")])
                    }
                    var s = a("free", []);
                    e.fileSize = e.lastDataIndex = c(n) + c(s);
                    i.File.open(e.downloadTarget + ".part", {
                        write: !0,
                        append: !1,
                        truncate: !0
                    }, {
                        unixMode: 420
                    }).then((function (e) {
                        f(e, [n, s], (function (n) {
                            e.close().catch((function () {})).then((function () {
                                t(n || null)
                            }))
                        }))
                    }), (function (e) {
                        t(e)
                    }))
                }, t.updateMdatLength = function (e, t, n, r) {
                    i.File.open(e.downloadTarget + ".part", {
                        write: !0,
                        append: !1
                    }).then((function (e) {
                        e.setPosition(t, i.File.POS_START).then((function () {
                            var t = new Uint8Array(4);
                            o.WriteInt32(t, 0, n + 8), e.write(t).then((function () {
                                e.close().catch((function () {})).then((function () {
                                    r(null)
                                }))
                            }), (function (t) {
                                e.close().catch((function () {})).then((function () {
                                    r(t)
                                }))
                            }))
                        }), (function (t) {
                            e.close().catch((function () {})).then((function () {
                                r(t)
                            }))
                        }))
                    }), (function (e) {
                        r(e)
                    }))
                };
                t.parse = p, t.getTags = h
            },
            226: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.getProxyHeaders = function (e) {
                    function t() {
                        var t = E[e];
                        t && (t.handlers.forEach((function (e) {
                            e.reject(new Error("timeout monitoring proxyHeaders"))
                        })), delete E[e])
                    }
                    var n = E[e];
                    n ? clearTimeout(n.timer) : n = E[e] = {
                        handlers: []
                    };
                    return new Promise((function (r, i) {
                        n.handlers.push({
                            resolve: r,
                            reject: i
                        }), n.timer = setTimeout(t, 3e4), fetch(e, {
                            method: "HEAD",
                            credentials: "include"
                        })
                    }))
                };
                var r = n(3),
                    i = n(12),
                    o = n(50),
                    a = n(7),
                    s = n(79),
                    u = n(217),
                    c = r.browser,
                    l = new RegExp("^bytes [0-9]+-[0-9]+/([0-9]+)$"),
                    d = new RegExp("^(audio|video)/(?:x-)?([^; ]+)"),
                    f = new RegExp('filename\\s*=\\s*"\\s*([^"]+?)\\s*"'),
                    p = new RegExp("/([^/]+?)(?:\\.([a-z0-9]{1,5}))?(?:\\?|#|$)", "i"),
                    h = new RegExp("\\.([a-z0-9]{1,5})(?:\\?|#|$)", "i"),
                    g = new RegExp("\\bsource=yt_otf\\b"),
                    m = new RegExp("/ptracking\\b"),
                    v = new RegExp("^https://www.gstatic.com/youtube/doodle\\b"),
                    y = new RegExp("^(https?)://v[^\\/]*\\.tumblr\\.com/(tumblr_[0-9a-zA-Z_]+)\\.(?:mp4|mov)"),
                    b = new RegExp("^https://soundcloud.com/"),
                    w = {},
                    k = {
                        host: !0,
                        range: !0,
                        "content-length": !0
                    };

                function x() {
                    w = {}, r.prefs.mediaExtensions.split("|").forEach((function (e) {
                        w[e] = 1
                    }))
                }
                x(), r.prefs.on("mediaExtensions", x);
                var A = null;

                function _() {
                    A = null;
                    var e = r.prefs.networkFilterOut;
                    if (e) try {
                        A = new RegExp(e, "i")
                    } catch (e) {
                        console.warn("networkFilterOut preference is not a valid regex")
                    }
                }
                _(), r.prefs.on("networkFilterOut", _);
                var O = function (e) {
                    var t = void 0;
                    if (P && (t = P[e.requestId]) && delete P[e.requestId], !(e.tabId < 0 || g.test(e.url) || m.test(e.url) || v.test(e.url) || A && A.test(e.url))) {
                        var n = {};
                        (e.responseHeaders || []).forEach((function (e) {
                            n[e.name.toLowerCase()] = e.value
                        }));
                        var x = n["content-type"],
                            _ = d.exec(x),
                            O = parseInt(n["content-length"]);
                        if (isNaN(O)) {
                            var I = n["content-range"];
                            if (I) {
                                var C = l.exec(I);
                                C && (O = parseInt(C[1]))
                            }
                        }
                        var S = h.exec(e.url),
                            j = null;
                        if (S) {
                            if ("m4s" == (j = S[1].toLowerCase()) && r.prefs.dashHideM4s) return;
                            if ("ts" == j && r.prefs.mpegtsHideTs) return;
                            w[S[1]] || (S = null)
                        }
                        var E = e.originUrl || e.documentUrl || void 0;
                        if (!(b.test(E) && O < 1e6 && "audio/mpeg" == x)) {
                            var D = u.networkHook(e, {
                                contentType: x,
                                referrer: E,
                                headers: t,
                                proxy: e.proxyInfo
                            });
                            if (D) D.then((function (e) {
                                return q(e, !0)
                            })).catch((function (e) {
                                console.error("Uncaught PostHook error:", e)
                            }));
                            else {
                                var T = _ && ("audio" == _[1] || "video" == _[1]),
                                    R = !isNaN(O) && r.prefs.mediaweightThreshold > 0 && O >= r.prefs.mediaweightThreshold;
                                q(null, T || R)
                            }
                        }
                    }

                    function q(u, l) {
                        if (l || !(!_ && isNaN(O) && !S || !_ && !S && (isNaN(O) || 0 === r.prefs.mediaweightThreshold || O < r.prefs.mediaweightThreshold) || _ && "ms-asf" == _[2].toLowerCase())) {
                            var d = {
                                id: "network-probe:" + a.hashHex(e.url),
                                status: "active",
                                url: e.url,
                                tabId: e.tabId,
                                frameId: e.frameId,
                                fromCache: !0,
                                referrer: E
                            };
                            isNaN(O) || (d.length = O), e.proxyInfo && "http" == e.proxyInfo.type.substr(0, 4) && (d.proxy = e.proxyInfo);
                            var h = n["content-disposition"];
                            if (h) {
                                var g = f.exec(h);
                                g && g[1] && (d.headerFilename = g[1])
                            }
                            var m = p.exec(e.url);
                            m && (d.urlFilename = m[1]), d.title = d.headerFilename || d.urlFilename || r._("media");
                            var v = y.exec(e.url);
                            v && (d.thumbnailUrl = v[1] + "://media.tumblr.com/" + v[2] + "_frame1.jpg"), S ? (d.type = "video", d.extension = S[1]) : _ ? (d.type = _[1], d.extension = _[2]) : d.extension = j, d.headers = t && t.filter((function (e) {
                                return void 0 === k[e.name.toLowerCase()]
                            })) || [], c.tabs.get(e.tabId).then((function (e) {
                                if (e) {
                                    d.topUrl = e.url, d.isPrivate = e.incognito, d.title = d.headerFilename || e.title || d.urlFilename || r._("media");
                                    var t = s.getSpecs(e.url);
                                    t && (t.headerFilename = d.headerFilename, t.urlFilename = d.urlFilename), a.executeScriptWithGlobal({
                                        tabId: e.id
                                    }, {
                                        _$vdhHitId: d.id,
                                        _$vdhSmartNameSpecs: t
                                    }, "/content/pagedata-script.js").catch((function (t) {
                                        c.webNavigation.getFrame({
                                            tabId: e.id,
                                            frameId: d.frameId
                                        }).then((function (n) {
                                            n && (console.warn("pagedata-script execution error", t.message), o.updateOriginal(d.id, {
                                                title: e.title || d.title,
                                                pageUrl: n.url,
                                                topUrl: e.url
                                            }))
                                        }))
                                    }))
                                }
                                u ? (d.originalId = d.id, u.handleHit(d)) : i.dispatch("hit.new", d)
                            }))
                        }
                    }
                };

                function I() {
                    c.webRequest.onHeadersReceived.addListener(O, {
                        urls: ["<all_urls>"]
                    }, ["responseHeaders"])
                }
                "firefox" == r.browserType && ["main_frame", "sub_frame", "xmlhttprequest", "object", "media"].push("object_subrequest"), r.prefs.on("networkProbe", (function (e, t) {
                    t ? I() : c.webRequest.onHeadersReceived.removeListener(O)
                })), r.prefs.networkProbe && I();
                var P = null,
                    C = function (e) {
                        D(e), P && (P[e.requestId] = e.requestHeaders)
                    },
                    S = function (e) {
                        D(e), P && delete P[e.requestId]
                    };

                function j() {
                    P = {}, c.webRequest.onSendHeaders.addListener(C, {
                        urls: ["<all_urls>"]
                    }, ["requestHeaders"]), c.webRequest.onErrorOccurred.addListener(S, {
                        urls: ["<all_urls>"]
                    })
                }
                r.prefs.on("monitorNetworkRequests", (function (e, t) {
                    t ? j() : (c.webRequest.onSendHeaders.removeListener(C), c.webRequest.onErrorOccurred.removeListener(S), P = null)
                })), r.prefs.monitorNetworkRequests && j();
                var E = {};

                function D(e) {
                    var t = E[e.url];
                    if (t) {
                        clearTimeout(t.timer), delete E[e.url];
                        var n = e.requestHeaders.filter((function (e) {
                            return void 0 === k[e.name.toLowerCase()]
                        }));
                        t.handlers.forEach((function (t) {
                            t.resolve({
                                proxy: e.proxyInfo,
                                headers: n
                            })
                        }))
                    }
                }
            },
            219: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = function () {
                    function e(e, t) {
                        for (var n = 0; n < t.length; n++) {
                            var r = t[n];
                            r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                        }
                    }
                    return function (t, n, r) {
                        return n && e(t.prototype, n), r && e(t, r), t
                    }
                }();
                var i = n(34),
                    o = n(7),
                    a = 2e5;
                t.File = function () {
                    function e(t) {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, e), this.fd = t, this.position = 0, this.fileLength = 0, this.writeControl = o.Concurrent(1)
                    }
                    return r(e, [{
                        key: "_close",
                        value: function () {
                            return i.call("fs.close", this.fd)
                        }
                    }, {
                        key: "_write",
                        value: function (e) {
                            var t = this;
                            return new Promise((function (n, r) {
                                var o = 0;
                                ! function s(u, c) {
                                    var l = e.subarray(u, c).toString();
                                    i.call("fs.write", t.fd, l, 0, c - u, t.position).then((function (r) {
                                        o += r, t.position += r, t.fileLength = Math.max(t.fileLength, t.position), o >= e.length ? n(o) : s(o, o + Math.min(e.length - o, a))
                                    })).catch(r)
                                }(0, Math.min(e.length, a))
                            }))
                        }
                    }, {
                        key: "_setPosition",
                        value: function (t) {
                            switch (arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : e.POS_START) {
                                case e.POS_START:
                                    this.position = t;
                                    break;
                                case e.POS_END:
                                    this.position = this.fileLength - t;
                                    break;
                                case e.POS_CUR:
                                    this.position += t
                            }
                            return Promise.resolve()
                        }
                    }, {
                        key: "write",
                        value: function () {
                            for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                            var r = this;
                            return r.writeControl((function () {
                                return r._write.apply(r, t)
                            }))
                        }
                    }, {
                        key: "close",
                        value: function () {
                            for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                            var r = this;
                            return r.writeControl((function () {
                                return r._close.apply(r, t)
                            }))
                        }
                    }, {
                        key: "setPosition",
                        value: function () {
                            for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                            var r = this;
                            return r.writeControl((function () {
                                return r._setPosition.apply(r, t)
                            }))
                        }
                    }], [{
                        key: "open",
                        value: function (t) {
                            var n = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                            return new Promise((function (r, o) {
                                var a = "r";
                                n.write && (a = "r+", n.truncate && (a = "w+")), i.call("fs.open", t, a).catch((function (e) {
                                    if ("r+" == a) return i.call("fs.open", t, "w").then((function (e) {
                                        return i.call("fs.close", e)
                                    })).then((function () {
                                        return i.call("fs.open", t, a)
                                    }));
                                    throw e
                                })).then((function (a) {
                                    var s = new e(a);
                                    n.truncate ? r(s) : i.call("fs.stat", t).then((function (e) {
                                        s.fileLength = e.size, s.position = e.size, r(s)
                                    })).catch(o)
                                })).catch(o)
                            }))
                        }
                    }, {
                        key: "move",
                        value: function () {
                            for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                            return i.call.apply(i, ["fs.rename"].concat(t))
                        }
                    }, {
                        key: "remove",
                        value: function () {
                            for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                            return i.call.apply(i, ["fs.unlink"].concat(t))
                        }
                    }, {
                        key: "POS_START",
                        get: function () {
                            return 0
                        }
                    }, {
                        key: "POS_CUR",
                        get: function () {
                            return 1
                        }
                    }, {
                        key: "POS_END",
                        get: function () {
                            return 2
                        }
                    }]), e
                }()
            },
            79: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.set = function (e) {
                    a = e, s()
                }, t.defineInPage = async function () {
                    var e = Math.round(1e9 * Math.random()),
                        t = await o.tabs.query({
                            active: !0,
                            currentWindow: !0
                        });
                    if (0 === t.length) throw new Error("Can't find current tab");
                    i.executeScriptWithGlobal({
                        tabId: t[0].id
                    }, {
                        _wehPanelName: "smartname-page-" + e
                    }, "/content/smartname-script.js")
                }, t.getSpecs = function (e) {
                    for (var t = new URL(e).hostname.split("."), n = 0; n < t.length - 1; n++) {
                        var r = a[t.slice(n).join(".")];
                        if (r) return r
                    }
                    return null
                }, t.getFilenameFromTitle = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : null,
                        n = {
                            keep: " ",
                            remove: "",
                            hyphen: "-",
                            underscore: "_"
                        };
                    t && (t = (t = t.replace(u, "")).replace(c, n[r.prefs.smartnamerFnameSpaces]));
                    e = (e = e.replace(u, "")).replace(c, n[r.prefs.smartnamerFnameSpaces]);
                    var i = r.prefs.smartnamerFnameMaxlen;
                    if (t) return e.length + t.length + 1 > i && (e = e.substr(0, i - t.length - 1)), e + "." + t;
                    e.length > i && (e = e.substr(0, i));
                    return e
                };
                var r = n(3),
                    i = n(7),
                    o = r.browser,
                    a = {};

                function s() {
                    o.storage.local.set({
                        smartname: a
                    }).catch((function (e) {
                        console.error("Cannot write smartname storage")
                    }))
                }
                o.storage.local.get({
                    smartname: {}
                }).then((function (e) {
                    a = e.smartname
                })).catch((function (e) {
                    console.error("Cannot read smartname storage")
                }));
                var u = new RegExp('[/?<>\\:*|":]|[\0--]|\\\\', "g"),
                    c = new RegExp(" +", "g");
                r.rpc.listen({
                    openSmartNameDefiner: function () {
                        return r.ui.open("smartname-definer", {
                            url: "content/smartname-define.html",
                            type: "panel",
                            width: 600,
                            height: 400
                        })
                    },
                    closeSmartNameDefiner: function () {
                        return r.ui.close("smartname-definer")
                    },
                    closedSmartNameDefiner: function (e) {
                        return r.rpc.call(e, "close")
                    },
                    setSmartNameData: function (e) {
                        return r.rpc.call("smartname-definer", "setData", e)
                    },
                    evaluateSmartName: function (e, t) {
                        return r.rpc.call(e, "evaluate", t)
                    },
                    addSmartNameRule: function (e) {
                        a[e.domain] = e, s()
                    },
                    selectSmartNameXPath: function (e, t) {
                        return r.rpc.call(e, "select", t)
                    },
                    setSmartName: function (e) {
                        a = {}, s()
                    },
                    getSmartNameRules: function () {
                        return a
                    },
                    editSmartName: function () {
                        r.ui.open("smartname-edit", {
                            type: "tab",
                            url: "content/smartname-edit.html"
                        })
                    },
                    removeFromSmartName: function (e) {
                        delete a[e], s()
                    }
                })
            },
            242: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = function () {
                        function e(e, t) {
                            for (var n = 0; n < t.length; n++) {
                                var r = t[n];
                                r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                            }
                        }
                        return function (t, n, r) {
                            return n && e(t.prototype, n), r && e(t, r), t
                        }
                    }(),
                    i = function e(t, n, r) {
                        null === t && (t = Function.prototype);
                        var i = Object.getOwnPropertyDescriptor(t, n);
                        if (void 0 === i) {
                            var o = Object.getPrototypeOf(t);
                            return null === o ? void 0 : e(o, n, r)
                        }
                        if ("value" in i) return i.value;
                        var a = i.get;
                        return void 0 !== a ? a.call(r) : void 0
                    };

                function o(e) {
                    if (Array.isArray(e)) {
                        for (var t = 0, n = Array(e.length); t < e.length; t++) n[t] = e[t];
                        return n
                    }
                    return Array.from(e)
                }

                function a(e, t) {
                    if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                }

                function s(e, t) {
                    if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                    return !t || "object" != typeof t && "function" != typeof t ? e : t
                }

                function u(e, t) {
                    if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                    e.prototype = Object.create(t && t.prototype, {
                        constructor: {
                            value: e,
                            enumerable: !1,
                            writable: !0,
                            configurable: !0
                        }
                    }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                }
                t.handleBulkVideo = function (e) {
                    var t = w[e.bulk];
                    if (!t) return void console.error("Bulk " + t.id + " does not exist");
                    var n = t.ids.indexOf(e.videoId);
                    if (n < 0) return void console.error("Video " + e.videoId + " is not part of bulk " + t.id);
                    return Object.assign(e, t.extra), t.ids.splice(n, 1), d.dispatch("hit.new", e), f.execute("quickdownload", e),
                        function (e) {
                            e.tabId && setTimeout((function () {
                                l.tabs.remove(e.tabId), e.tabId = 0
                            }), 0);
                            var t = e.resolve;
                            if (delete e.resolve, delete e.reject, e.count > 0) {
                                var n = Math.round(100 * (e.count - e.ids.length) / e.count);
                                p.updateProgress(e.id, n)
                            }
                            if (0 === e.ids.length || e.aborted) {
                                var r = e.actionResolve;
                                r && (delete e.actionResolve, r())
                            }
                            t()
                        }(t), !0
                };
                var c = n(3),
                    l = c.browser,
                    d = n(12),
                    f = n(189),
                    p = n(50),
                    h = n(7),
                    g = n(196),
                    m = n(192),
                    v = n(8).buildOptions.noyt || !1,
                    y = "youtube",
                    b = h.Concurrent(),
                    w = {};

                function k(e) {
                    e.ids.forEach((function (t) {
                        ! function (e, t) {
                            b((function () {
                                return function (e, t) {
                                    var n = new Promise((function (t, n) {
                                            e.resolve = t, e.reject = n
                                        })),
                                        r = "https://www." + y + ".com/watch?v=" + t + "&vdh-bulk=" + e.id;
                                    return l.tabs.create({
                                        url: r,
                                        active: !1
                                    }).then((function (t) {
                                        return e.tabId = t.id, l.tabs.update(t.id, {
                                            muted: !0
                                        }), n
                                    }))
                                }(e, t)
                            })).catch((function (e) {
                                console.error("BulkDownload !!!", e.message)
                            }))
                        }(e, t)
                    }))
                }
                c.rpc.listen({
                    tbvwsSelectedIds: function (e) {
                        var t = [],
                            n = d.getHits();
                        if (Object.keys(n).forEach((function (r) {
                                var i = n[r];
                                "tbvws-bulk" != i.from || i.topUrl != e.topUrl || i.running || t.push(i.id)
                            })), t.length > 0 && d.dispatch("hit.delete", t), e.ids.length > 0) {
                            var r = {
                                id: "tbvws-bulk:" + Math.floor(1e9 * Math.random()),
                                title: c._("selected_media"),
                                descrPrefix: c._("bulk_n_videos", "" + e.ids.length),
                                from: "tbvws-bulk",
                                ids: e.ids,
                                pageUrl: e.pageUrl,
                                topUrl: e.topUrl,
                                thumbnailUrl: l.runtime.getURL("/content/images/tbvws.png")
                            };
                            d.dispatch("hit.new", r)
                        }
                    }
                }), setTimeout((function () {
                    var e = function (e) {
                        function t() {
                            return a(this, t), s(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return u(t, e), r(t, [{
                            key: "doJob",
                            value: function () {
                                var e = this;
                                p.update(this.hit.id, {
                                    operation: "collecting",
                                    opStartDate: Date.now()
                                });
                                var t = {
                                    id: this.hit.id,
                                    aborted: !1,
                                    count: this.hit.ids.length,
                                    ids: [].concat(o(this.hit.ids)),
                                    tabId: null,
                                    resolve: null,
                                    reject: null,
                                    extra: e.bulkExtra || void 0
                                };
                                w[this.hit.id] = t;
                                var n = new Promise((function (n, r) {
                                    t.actionResolve = n, t.actionReject = r, e.setAbort((function () {
                                        t.aborted = !0, r(new h.VDHError("Aborted", {
                                            noReport: !0
                                        }))
                                    }))
                                }));
                                return k(t), n
                            }
                        }, {
                            key: "getReqs",
                            value: function () {
                                if (v && g.matchHit(this.hit)) return g.forbidden(), Promise.reject(new h.VDHError("Forbidden", {
                                    noReport: !0
                                }));
                                this.reqs.coapp = !0, this.reqs.coappMin = "1.6.2"
                            }
                        }, {
                            key: "postJob",
                            value: function () {}
                        }, {
                            key: "cleanup",
                            value: function () {
                                return this.clearAbort(), d.dispatch("hit.delete", this.hit.id), i(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "cleanup", this).call(this)
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !(e.running > 0) && "tbvws-bulk" == e.from
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "bulkdownload"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return c._("action_bulkdownload_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return c._("action_bulkdownload_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-quick-download2-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 300
                            }
                        }, {
                            key: "catPriority",
                            get: function () {
                                return 2
                            }
                        }]), t
                    }(f.DownloadAction);
                    f.register(e);
                    var t = function (e) {
                        function t() {
                            return a(this, t), s(this, (t.__proto__ || Object.getPrototypeOf(t)).apply(this, arguments))
                        }
                        return u(t, e), r(t, [{
                            key: "getReqs",
                            value: function () {
                                var e = this,
                                    n = this;
                                return Promise.resolve().then((function () {
                                    var t = "dlconv#" + e.hit.id;
                                    return c.openedContents().indexOf("main") >= 0 ? c.rpc.call("main", "embed", l.runtime.getURL("content/dlconv-embed.html?nosaveas=1&panel=" + t)).then((function () {
                                        return c.wait(t)
                                    })).catch((function (e) {
                                        throw new h.VDHError("Aborted", {
                                            noReport: !0
                                        })
                                    })) : m.getOutputConfigs().then((function (e) {
                                        var t = c.prefs.dlconvLastOutput || "05cb6b27-9167-4d83-833d-218a107d0376",
                                            n = e[t];
                                        if (!n) throw new Error("No such output configuration");
                                        return {
                                            outputConfigId: t,
                                            outputConfig: n
                                        }
                                    }))
                                })).then((function (e) {
                                    c.prefs.dlconvLastOutput = e.outputConfigId;
                                    var t = e.outputConfig;
                                    n.bulkExtra = {
                                        convert: t
                                    };
                                    var r = t.ext || t.params.f;
                                    r && (n.bulkExtra.extension = r)
                                })).then((function () {
                                    return i(t.prototype.__proto__ || Object.getPrototypeOf(t.prototype), "getReqs", e).call(e)
                                }))
                            }
                        }], [{
                            key: "canPerform",
                            value: function (e) {
                                return !(e.running > 0) && "tbvws-bulk" == e.from
                            }
                        }, {
                            key: "name",
                            get: function () {
                                return "bulkdownloadconvert"
                            }
                        }, {
                            key: "title",
                            get: function () {
                                return c._("action_bulkdownloadconvert_title")
                            }
                        }, {
                            key: "description",
                            get: function () {
                                return c._("action_bulkdownloadconvert_description")
                            }
                        }, {
                            key: "icon",
                            get: function () {
                                return "images/icon-action-download-convert-64.png"
                            }
                        }, {
                            key: "priority",
                            get: function () {
                                return 80
                            }
                        }, {
                            key: "keepOpen",
                            get: function () {
                                return !0
                            }
                        }]), t
                    }(e);
                    f.register(t)
                }), 0)
            },
            196: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.matchHit = function (e) {
                    return ![e.url, e.audioUrl, e.videoUrl, e.pageUrl, e.topUrl].every((function (e) {
                        return !h.test(e) && !p.test(e)
                    }))
                }, t.forbidden = function () {
                    var e = r._("chrome_noyt_text"),
                        t = c.hash(e),
                        n = r._("chrome_noyt_text3"),
                        i = c.hash(n),
                        o = n; - 1960581238 == i && -1126813505 != t && (o = e);
                    d.alert({
                        title: r._("chrome_warning_yt"),
                        text: [o, r._("chrome_noyt_text2")],
                        height: 400,
                        buttons: [{
                            text: r._("chrome_install_firefox"),
                            className: "btn-outline-secondary",
                            close: !0,
                            trigger: {
                                what: "installFirefox"
                            }
                        }, {
                            text: r._("chrome_install_fx_vdh"),
                            className: "btn-outline-primary",
                            close: !0,
                            trigger: {
                                what: "vdhForFirefox"
                            }
                        }]
                    }).then((function (e) {
                        switch (e.what) {
                            case "installFirefox":
                                return l.gotoOrOpenTab("https://getfirefox.com/");
                            case "vdhForFirefox":
                                return l.gotoOrOpenTab("https://addons.mozilla.org/firefox/addon/video-downloadhelper/")
                        }
                    })).catch((function (e) {
                        console.error("tbvws error", e)
                    }))
                };
                var r = n(3),
                    i = r.browser,
                    o = n(12),
                    a = n(222),
                    s = n(79),
                    u = n(242),
                    c = n(7),
                    l = n(49),
                    d = n(194),
                    f = new RegExp("^https?://([^/]*\\.)?youtube(\\.co)?.([^./]+)/"),
                    p = new RegExp("^https?://([^/]*.)?googlevideo\\."),
                    h = new RegExp("^https?://([^/]*\\.)?youtube(\\.co)?.([^./]+)/.*");

                function g(e) {
                    var t = a.getHitsFromVariants(Object.assign(function (e) {
                        return {
                            id: "tbvws:" + e.videoId,
                            group: "tbvws:" + e.videoId,
                            tabId: e.tabId,
                            title: e.title,
                            from: "tbvws",
                            videoId: e.videoId,
                            topUrl: e.topUrl,
                            pageUrl: e.pageUrl,
                            thumbnailUrl: e.videoDetails.thumbnail.thumbnails[0].url,
                            duration: parseInt(e.videoDetails.lengthSeconds) || void 0,
                            headers: [],
                            proxy: null
                        }
                    }(e), {}), e.formats.concat(e.adaptiveFormats), {
                        audioAndVideo: !0,
                        keepProtected: !0
                    });
                    if (t = t.filter((function (e) {
                            if (void 0 === e.url) return !0;
                            var t = new RegExp("^d+x(d+)$").exec(e.size || "");
                            if (t) {
                                var n = parseInt(t[1]);
                                return !isNaN(n && n <= 360)
                            }
                        })), e.bulk) {
                        var n = t[0];
                        n.bulk = e.bulk, u.handleBulkVideo(n)
                    } else t.forEach((function (e) {
                        o.dispatch("hit.new", e)
                    }))
                }
                var m = {
                    url: [{
                        hostContains: "youtube"
                    }]
                };
                i.webNavigation.onCompleted.addListener((function (e) {
                    var t = {
                        tabId: e.tabId,
                        frameIds: [e.frameId]
                    };
                    f.test(e.url) && i.tabs.get(e.tabId).then((function (n) {
                        c.executeScriptWithGlobal(t, {
                            _$vdhData: t,
                            _$vdhSmartNameSpecs: s.getSpecs(e.url),
                            _$vdhTopUrl: n.url,
                            _$vdhExtractMethod: r.prefs.tbvwsExtractionMethod
                        }, "/content/tbvws-script.js")
                    })).catch((function (e) {
                        console.error("Cannot find tab", e)
                    })), r.prefs.bulkEnabled && e.frameId > 0 && h.test(e.url) && i.tabs.get(e.tabId).then((function (e) {
                        c.executeScriptWithGlobal(t, {
                            _$vdhTopUrl: e.url
                        }, "/content/tbvws-bulk-script.js").catch((function (e) {
                            console.error("VDH error: could not inject bulk script", e)
                        }))
                    }))
                }), m), r.rpc.listen({
                    tbvwsDetectedVideo: function (e) {
                        try {
                            g(e)
                        } catch (e) {
                            console.error("VDH error: detectedVideo", e)
                        }
                    }
                })
            },
            222: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.getHitsFromVariants = function (e, t, n) {
                    n = n || {};
                    var i = {},
                        s = {},
                        u = r.prefs.ignoreProtectedVariants,
                        c = !0;
                    t.forEach((function (t) {
                        if (t.url && !(u && !n.keepProtected && t.s && t.s.length > 0)) {
                            i[t.itag] = t;
                            var a = e.from + ":" + t.itag,
                                d = "audio/video",
                                f = null,
                                p = null,
                                h = null,
                                g = l.variants.full[a];
                            if (g) {
                                if (!g.enabled) return;
                                g.audioCodec && (f = g.audioCodec, d = "audio"), g.videoCodec && (p = g.videoCodec, d = f ? "audio/video" : "video"), h = g.extension
                            } else if (void 0 !== t.type || void 0 !== t.mimeType) {
                                var m = o.exec(t.type || t.mimeType);
                                if (m)
                                    if (h = m[2], "audio" == m[1]) d = "audio", f = m[3] || null, /vorbis|opus/i.test(f) && (h = "webm");
                                    else if (m[3]) {
                                    var v = m[3].split(",");
                                    1 == v.length ? (d = "video", p = v[0], /vp8|vp9/i.test(p) && (h = "webm")) : (p = v[0], f = v[1])
                                } else "video" == m[1] && (d = "video")
                            }
                            if (h = h || "mp4", "audio/video" === d) {
                                var y = {
                                    id: e.from + ":" + e.videoId + ":" + t.itag,
                                    url: t.url,
                                    extension: h
                                };
                                t.quality && (y.quality = t.quality), f && (y.audioCodec = f), p && (y.videoCodec = p), t.size && (y.size = t.size), t.fps && (y.fps = t.fps), t.s && (y._signature = t.s), s[a] = y;
                                var b = l.variants.full[a];
                                if (void 0 === b) {
                                    l.variants.full[a] = {
                                        extension: h,
                                        quality: t.quality,
                                        audioCodec: f,
                                        videoCodec: p,
                                        enabled: !0
                                    }, c = !0;
                                    var w = [];
                                    if (w.push("(" + a + ")"), t.quality) {
                                        var k = "quality_" + t.quality,
                                            x = r._(k);
                                        x == k && (x = t.quality.toUpperCase()), w.push(x)
                                    }
                                    t.size && w.push(t.size), w.push(h.toUpperCase()), p && w.push("V/" + p), f && w.push("A/" + f), l.variants.full[a].label = w.join(" - "), l.variants.full_list.push(a)
                                } else !t.size && b.size && (y.size = b.size)
                            } else {
                                var A = void 0,
                                    _ = void 0;
                                if ("audio" == d ? (A = "adp_audio", _ = "adp_video") : (A = "adp_video", _ = "adp_audio"), void 0 === l.variants[A][t.itag]) {
                                    var O = {
                                        extension: h
                                    };
                                    for (var I in t.size ? O.size = t.size : t.width && t.height && (O.size = t.width + "x" + t.height), t.bitrate && (O.bitRate = t.bitrate), t.fps && (O.fps = t.fps), c = !0, l.variants[A][t.itag] = O, l.variants[_]) "audio" == d ? l.variants.adp_list.push(I + "/" + t.itag) : l.variants.adp_list.push(t.itag + "/" + I);
                                    "audio" == d ? (l.variants[A][t.itag].codec = f, l.variants.adp_list.push("/" + t.itag)) : (l.variants[A][t.itag].codec = p, l.variants.adp_list.push(t.itag + "/"))
                                }
                            }
                        }
                    }));
                    for (var f = e.maxVariants || r.prefs.qualitiesMaxVariants, p = 0, h = [], g = null, m = 0; m < l.variants.full_list.length && (!f || p < f); m++) {
                        var v = l.variants.full_list[m],
                            y = /^adp:([0-9]+)$/.exec(v);
                        if (y) {
                            if (r.prefs.adpHide) continue;
                            g || (g = [], l.variants.adp_list.forEach((function (t) {
                                if (!n.audioAndVideo || /^\d+\/\d+$/.test(t)) {
                                    var r = a.exec(t);
                                    if ((!(r[1].length > 0) || i[r[1]]) && (!(r[1].length > 0 && l.variants.adp_video[r[1]] && "vp9" == l.variants.adp_video[r[1]].codec && l.converter) || l.converter.vp9support) && (!(r[2].length > 0) || i[r[2]])) {
                                        var o = i[r[1]],
                                            s = i[r[2]],
                                            u = {
                                                id: e.from + ":" + e.videoId + ":" + t,
                                                _signature: []
                                            },
                                            c = 0,
                                            d = null,
                                            f = null;
                                        o && (u.videoUrl = o.url, o.clen && (c += parseInt(o.clen)), s || (u.url = u.videoUrl), o.s && u._signature.push(o.s), d = l.variants.adp_video[r[1]], u.extension = o.extension || (d ? d.extension : void 0), o.size && (u.size = o.size), !u.size && d && d.size && (u.size = d.size), o.fps && (u.fps = o.fps), !u.fps && d && d.fps && (u.fps = d.fps)), s && (f = l.variants.adp_audio[r[2]], u.audioUrl = s.url, s.clen && (c += parseInt(s.clen)), o || (u.url = u.audioUrl, u.extension = s.extension), s.s && u._signature.push(s.s)), f && d && f.extension != d.extension && (u.extension = "mkv"), c && (u.length = c), u.group = e.group, g.push(u)
                                    }
                                }
                            })));
                            var b = parseInt(y[1]) - 1;
                            b < g.length && (h[p++] = Object.assign({
                                order: p,
                                adp: !0
                            }, e, g[b]))
                        } else s[v] && (h[p++] = Object.assign({
                            order: p
                        }, e, s[v]))
                    }
                    c && d();
                    return h
                }, t.getVariantsList = f, t.setVariantsList = p, t.setVariants = function (e) {
                    l.variants = e, d()
                }, t.getAdpVariantsList = h, t.setAdpVariantsList = g, t.orderAdaptative = function () {
                    l.variants.adp_list.sort((function (e, t) {
                        var n = s.exec(e),
                            r = s.exec(t),
                            i = l.variants.adp_video[n[1]],
                            o = l.variants.adp_video[r[1]],
                            a = l.variants.adp_audio[n[2]],
                            u = l.variants.adp_audio[r[2]],
                            c = i && parseInt(i.size) || 0;
                        return (o && parseInt(o.size) || 0) * (u ? 1 : 0) - c * (a ? 1 : 0)
                    }))
                }, t.hasAudioVideo = function (e) {
                    var t = {
                            audio: !1,
                            video: !1
                        },
                        n = l.variants.full[e];
                    n && (t.audio = !!n.audioCodec, t.video = !!n.videoCodec);
                    return t
                };
                var r = n(3),
                    i = r.browser,
                    o = new RegExp('^(audio|video)/(?:x\\-)?([^;]+)(?:;(?:\\+| )codecs="(.+)")?$'),
                    a = new RegExp("^([0-9]*)/([0-9]*)$"),
                    s = new RegExp("^(.*)/(.*)$"),
                    u = new RegExp("^adp:[0-9]+$");
                r.rpc.listen({
                    editVariants: function () {
                        r.ui.open("variants-edit", {
                            type: "tab",
                            url: "content/variants-edit.html"
                        })
                    },
                    getVariantsLists: function () {
                        return {
                            full: f(),
                            adp: h()
                        }
                    },
                    setVariantsLists: function (e) {
                        return e.full && p(e.full), e.adp && g(e.adp), d()
                    },
                    resetVariants: function () {
                        return l.variants = c, d()
                    }
                });
                var c = {
                        full: {
                            "tbvws:22": {
                                extension: "mp4",
                                quality: "hd720",
                                audioCodec: "+mp4a.40.2",
                                videoCodec: "avc1.64001F",
                                enabled: !0,
                                label: "MP4 - 1280x720",
                                size: "1280x720"
                            },
                            "tbvws:18": {
                                extension: "mp4",
                                quality: "medium",
                                audioCodec: "+mp4a.40.2",
                                videoCodec: "avc1.42001E",
                                enabled: !0,
                                label: "MP4 - 480x360",
                                size: "480x360"
                            },
                            "tbvws:43": {
                                extension: "webm",
                                quality: "medium",
                                audioCodec: "+vorbis",
                                videoCodec: "vp8.0",
                                enabled: !0,
                                label: "WEBM - 480x360",
                                size: "480x360"
                            },
                            "tbvws:5": {
                                extension: "flv",
                                quality: "small",
                                audioCodec: null,
                                videoCodec: null,
                                enabled: !0,
                                label: "FLV - 320x240",
                                size: "320x240"
                            },
                            "tbvws:36": {
                                extension: "3gpp",
                                quality: "small",
                                audioCodec: "+mp4a.40.2",
                                videoCodec: "mp4v.20.3",
                                enabled: !0,
                                label: "3GPP - 320x240",
                                size: "320x240"
                            },
                            "tbvws:17": {
                                extension: "3gpp",
                                quality: "small",
                                audioCodec: "+mp4a.40.2",
                                videoCodec: "mp4v.20.3",
                                enabled: !0,
                                label: "3GPP - 176x144",
                                size: "176x144"
                            },
                            "tfvws:1080-0": {
                                extension: "mp4",
                                audioCodec: null,
                                videoCodec: null,
                                enabled: !0,
                                size: "1920x1080",
                                label: "H264_1920x1080 -MP4"
                            },
                            "tfvws:720-0": {
                                extension: "mp4",
                                audioCodec: null,
                                videoCodec: null,
                                enabled: !0,
                                size: "1280x720",
                                label: "H264_1280x720 -MP4"
                            },
                            "tfvws:480-0": {
                                extension: "mp4",
                                audioCodec: null,
                                videoCodec: null,
                                enabled: !0,
                                size: "848x480",
                                label: "H264_848x480 -MP4"
                            },
                            "tfvws:380-0": {
                                extension: "mp4",
                                audioCodec: null,
                                videoCodec: null,
                                enabled: !0,
                                size: "512x384",
                                label: "H264_512x384 -MP4"
                            },
                            "tfvws:240-0": {
                                extension: "mp4",
                                audioCodec: null,
                                videoCodec: null,
                                enabled: !0,
                                size: "320x240",
                                label: "H264_320x240 -MP4"
                            }
                        },
                        full_list: ["adp:1", "adp:2", "adp:3", "adp:4", "adp:5", "adp:6", "adp:7", "adp:8", "adp:9", "adp:10", "adp:11", "adp:12", "tbvws:22", "tbvws:18", "tbvws:5", "tbvws:17", "tbvws:43", "tbvws:36", "tfvws:1080-0", "tfvws:720-0", "tfvws:480-0", "tfvws:380-0", "tfvws:240-0"],
                        adp_audio: {
                            139: {
                                extension: "mp4",
                                bitRate: 50013,
                                codec: "mp4a.40.5"
                            },
                            140: {
                                extension: "mp4",
                                bitRate: 130535,
                                codec: "mp4a.40.2"
                            },
                            249: {
                                extension: "webm",
                                bitRate: 57181,
                                codec: "opus"
                            },
                            250: {
                                extension: "webm",
                                bitRate: 75052,
                                codec: "opus"
                            },
                            251: {
                                extension: "webm",
                                bitRate: 148416,
                                codec: "opus"
                            },
                            599: {
                                extension: "mp4",
                                bitRate: 32122,
                                codec: "mp4a.40.5"
                            },
                            600: {
                                extension: "webm",
                                bitRate: 40545,
                                codec: "opus"
                            }
                        },
                        adp_video: {
                            133: {
                                extension: "mp4",
                                size: "426x240",
                                bitRate: 245321,
                                fps: 30,
                                codec: "avc1.4d4015"
                            },
                            134: {
                                extension: "mp4",
                                size: "640x360",
                                bitRate: 633553,
                                fps: 30,
                                codec: "avc1.4d401e"
                            },
                            135: {
                                extension: "mp4",
                                size: "854x480",
                                bitRate: 1158055,
                                fps: 30,
                                codec: "avc1.4d401f"
                            },
                            136: {
                                extension: "mp4",
                                size: "1280x720",
                                bitRate: 1692034,
                                fps: 30,
                                codec: "avc1.64001f"
                            },
                            137: {
                                extension: "mp4",
                                size: "1920x1080",
                                bitRate: 2900516,
                                fps: 30,
                                codec: "avc1.640028"
                            },
                            160: {
                                extension: "mp4",
                                size: "256x144",
                                bitRate: 111180,
                                fps: 30,
                                codec: "avc1.4d400c"
                            },
                            242: {
                                extension: "webm",
                                size: "426x240",
                                bitRate: 198638,
                                fps: 30,
                                codec: "vp9"
                            },
                            243: {
                                extension: "webm",
                                size: "640x360",
                                bitRate: 433020,
                                fps: 30,
                                codec: "vp9"
                            },
                            244: {
                                extension: "webm",
                                size: "854x480",
                                bitRate: 856873,
                                fps: 30,
                                codec: "vp9"
                            },
                            247: {
                                extension: "webm",
                                size: "1280x720",
                                bitRate: 820432,
                                fps: 30,
                                codec: "vp9"
                            },
                            248: {
                                extension: "webm",
                                size: "1920x1080",
                                bitRate: 1590337,
                                fps: 30,
                                codec: "vp9"
                            },
                            271: {
                                extension: "webm",
                                size: "2560x1048",
                                bitRate: 6369110,
                                fps: 25,
                                codec: "vp9"
                            },
                            278: {
                                extension: "webm",
                                size: "256x144",
                                bitRate: 90329,
                                fps: 30,
                                codec: "vp9"
                            },
                            298: {
                                extension: "mp4",
                                size: "1280x720",
                                bitRate: 3483484,
                                fps: 60,
                                codec: "avc1.4d4020"
                            },
                            299: {
                                extension: "mp4",
                                size: "1920x1080",
                                bitRate: 5794998,
                                fps: 60,
                                codec: "avc1.64002a"
                            },
                            302: {
                                extension: "webm",
                                size: "1280x720",
                                bitRate: 2669978,
                                fps: 60,
                                codec: "vp9"
                            },
                            303: {
                                extension: "webm",
                                size: "1920x1080",
                                bitRate: 4620081,
                                fps: 60,
                                codec: "vp9"
                            },
                            308: {
                                extension: "webm",
                                size: "2560x1440",
                                bitRate: 13327223,
                                fps: 60,
                                codec: "vp9"
                            },
                            313: {
                                extension: "webm",
                                size: "3840x1574",
                                bitRate: 12855825,
                                fps: 25,
                                codec: "vp9"
                            },
                            315: {
                                extension: "webm",
                                size: "3840x2160",
                                bitRate: 26650187,
                                fps: 60,
                                codec: "vp9"
                            },
                            330: {
                                extension: "webm",
                                size: "256x144",
                                bitRate: 244830,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            331: {
                                extension: "webm",
                                size: "426x240",
                                bitRate: 500446,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            332: {
                                extension: "webm",
                                size: "640x360",
                                bitRate: 1060554,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            333: {
                                extension: "webm",
                                size: "854x480",
                                bitRate: 1989046,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            334: {
                                extension: "webm",
                                size: "1280x720",
                                bitRate: 4529350,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            335: {
                                extension: "webm",
                                size: "1920x1080",
                                bitRate: 6956417,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            336: {
                                extension: "webm",
                                size: "2560x1440",
                                bitRate: 16904531,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            337: {
                                extension: "webm",
                                size: "3840x2160",
                                bitRate: 30616762,
                                fps: 60,
                                codec: "vp9.2"
                            },
                            394: {
                                extension: "mp4",
                                size: "256x144",
                                bitRate: 88099,
                                fps: 30,
                                codec: "av01.0.00M.10.0.110.09.16.09.0"
                            },
                            395: {
                                extension: "mp4",
                                size: "426x240",
                                bitRate: 192111,
                                fps: 30,
                                codec: "av01.0.00M.10.0.110.09.16.09.0"
                            },
                            396: {
                                extension: "mp4",
                                size: "640x360",
                                bitRate: 407864,
                                fps: 30,
                                codec: "av01.0.01M.10.0.110.09.16.09.0"
                            },
                            397: {
                                extension: "mp4",
                                size: "854x480",
                                bitRate: 764544,
                                fps: 30,
                                codec: "av01.0.04M.10.0.110.09.16.09.0"
                            },
                            398: {
                                extension: "mp4",
                                size: "1280x720",
                                bitRate: 2100280,
                                fps: 60,
                                codec: "av01.0.08M.10.0.110.09.16.09.0"
                            },
                            399: {
                                extension: "mp4",
                                size: "1920x1080",
                                bitRate: 3817449,
                                fps: 60,
                                codec: "av01.0.09M.10.0.110.09.16.09.0"
                            },
                            400: {
                                extension: "mp4",
                                size: "2560x1440",
                                bitRate: 8580436,
                                fps: 60,
                                codec: "av01.0.12M.10.0.110.09.16.09.0"
                            },
                            401: {
                                extension: "mp4",
                                size: "3840x2160",
                                bitRate: 17186067,
                                fps: 60,
                                codec: "av01.0.13M.10.0.110.09.16.09.0"
                            },
                            597: {
                                extension: "mp4",
                                size: "256x144",
                                bitRate: 38793,
                                fps: 15,
                                codec: "avc1.4d400b"
                            },
                            598: {
                                extension: "webm",
                                size: "256x144",
                                bitRate: 34801,
                                fps: 15,
                                codec: "vp9"
                            },
                            694: {
                                extension: "mp4",
                                size: "256x144",
                                bitRate: 188378,
                                fps: 60,
                                codec: "av01.0.00M.10.0.110.09.16.09.0"
                            },
                            695: {
                                extension: "mp4",
                                size: "426x240",
                                bitRate: 397324,
                                fps: 60,
                                codec: "av01.0.01M.10.0.110.09.16.09.0"
                            },
                            696: {
                                extension: "mp4",
                                size: "640x360",
                                bitRate: 837578,
                                fps: 60,
                                codec: "av01.0.04M.10.0.110.09.16.09.0"
                            },
                            697: {
                                extension: "mp4",
                                size: "854x480",
                                bitRate: 1502575,
                                fps: 60,
                                codec: "av01.0.05M.10.0.110.09.16.09.0"
                            },
                            698: {
                                extension: "mp4",
                                size: "1280x720",
                                bitRate: 3871764,
                                fps: 60,
                                codec: "av01.0.08M.10.0.110.09.16.09.0"
                            },
                            699: {
                                extension: "mp4",
                                size: "1920x1080",
                                bitRate: 6531652,
                                fps: 60,
                                codec: "av01.0.09M.10.0.110.09.16.09.0"
                            },
                            700: {
                                extension: "mp4",
                                size: "2560x1440",
                                bitRate: 18487828,
                                fps: 60,
                                codec: "av01.0.12M.10.0.110.09.16.09.0"
                            },
                            701: {
                                extension: "mp4",
                                size: "3840x2160",
                                bitRate: 32652217,
                                fps: 60,
                                codec: "av01.0.13M.10.0.110.09.16.09.0"
                            }
                        },
                        adp_list: ["401/140", "400/140", "399/140", "298/140", "135/140", "315/251", "336/251", "335/251", "334/251", "401/251", "313/251", "271/251", "266/140", "264/140", "137/140", "136/140", "266/171", "264/171", "137/171", "136/171", "248/140", "248/171", "299/171", "299/140", "303/140", "303/171", "247/140", "247/171", "298/171", "302/140", "302/171", "135/171", "244/140", "244/171", "134/140", "134/171", "133/140", "243/171", "242/140", "243/140", "242/171", "133/171", "278/140", "278/171", "160/140", "160/171", "264/", "133/", "242/", "243/", "160/", "/140", "/171", "136/", "247/", "135/", "244/", "134/", "137/", "248/", "278/", "299/", "303/", "298/", "302/", "266/", "313/140", "313/171", "313/", "271/140", "271/171", "271/", "133/251", "134/251", "135/251", "136/251", "137/251", "160/251", "242/251", "243/251", "244/251", "247/251", "248/251", "264/251", "266/251", "278/251", "298/251", "299/251", "302/251", "303/251", "/251", "397/139", "397/140", "397/171", "397/251", "397/", "395/139", "395/140", "395/171", "395/251", "395/", "133/250", "134/250", "135/250", "136/250", "137/250", "160/250", "242/250", "243/250", "244/250", "247/250", "248/250", "264/250", "266/250", "271/250", "278/250", "298/250", "299/250", "302/250", "303/250", "313/250", "395/250", "397/250", "/250", "396/139", "396/140", "396/171", "396/250", "396/251", "396/", "133/249", "134/249", "135/249", "136/249", "137/249", "160/249", "242/249", "243/249", "244/249", "247/249", "248/249", "264/249", "266/249", "271/249", "278/249", "298/249", "299/249", "302/249", "303/249", "313/249", "395/249", "396/249", "397/249", "/249", "315/140", "315/139", "315/171", "315/249", "315/250", "315/", "337/139", "337/140", "337/171", "337/249", "337/250", "337/251", "337/", "401/139", "401/171", "401/249", "401/250", "401/", "701/139", "701/140", "701/171", "701/249", "701/250", "701/251", "701/", "308/139", "308/140", "308/171", "308/249", "308/250", "308/251", "308/", "336/139", "336/140", "336/171", "336/249", "336/250", "336/", "400/139", "400/171", "400/249", "400/250", "400/251", "400/", "700/139", "700/140", "700/171", "700/249", "700/250", "700/251", "700/", "335/139", "335/140", "335/171", "335/249", "335/250", "335/", "399/139", "399/171", "399/249", "399/250", "399/251", "399/", "699/139", "699/140", "699/171", "699/249", "699/250", "699/251", "699/", "334/139", "334/140", "334/171", "334/249", "334/250", "334/", "398/139", "398/140", "398/171", "398/249", "398/250", "398/251", "398/", "698/139", "698/140", "698/171", "698/249", "698/250", "698/251", "698/", "333/139", "333/140", "333/171", "333/249", "333/250", "333/251", "333/", "697/139", "697/140", "697/171", "697/249", "697/250", "697/251", "697/", "332/139", "332/140", "332/171", "332/249", "332/250", "332/251", "332/", "696/139", "696/140", "696/171", "696/249", "696/250", "696/251", "696/", "331/139", "331/140", "331/171", "331/249", "331/250", "331/251", "331/", "695/139", "695/140", "695/171", "695/249", "695/250", "695/251", "695/", "330/139", "330/140", "330/171", "330/249", "330/250", "330/251", "330/", "394/139", "394/140", "394/171", "394/249", "394/250", "394/251", "394/", "597/139", "597/140", "597/171", "597/249", "597/250", "597/251", "597/", "598/139", "598/140", "598/171", "598/249", "598/250", "598/251", "598/", "694/139", "694/140", "694/171", "694/249", "694/250", "694/251", "694/", "133/599", "134/599", "135/599", "136/599", "137/599", "160/599", "242/599", "243/599", "244/599", "247/599", "248/599", "264/599", "266/599", "271/599", "278/599", "298/599", "299/599", "302/599", "303/599", "308/599", "313/599", "315/599", "330/599", "331/599", "332/599", "333/599", "334/599", "335/599", "336/599", "337/599", "394/599", "395/599", "396/599", "397/599", "398/599", "399/599", "400/599", "401/599", "597/599", "598/599", "694/599", "695/599", "696/599", "697/599", "698/599", "699/599", "700/599", "701/599", "/599", "133/600", "134/600", "135/600", "136/600", "137/600", "160/600", "242/600", "243/600", "244/600", "247/600", "248/600", "264/600", "266/600", "271/600", "278/600", "298/600", "299/600", "302/600", "303/600", "308/600", "313/600", "315/600", "330/600", "331/600", "332/600", "333/600", "334/600", "335/600", "336/600", "337/600", "394/600", "395/600", "396/600", "397/600", "398/600", "399/600", "400/600", "401/600", "597/600", "598/600", "694/600", "695/600", "696/600", "697/600", "698/600", "699/600", "700/600", "701/600", "/600", "315/", "337/", "401/", "701/", "308/", "336/", "400/", "700/", "299/", "303/", "335/", "399/", "699/", "298/", "302/", "334/", "398/", "698/", "135/", "244/", "333/", "397/", "697/", "134/", "243/", "332/", "396/", "696/", "133/", "242/", "331/", "395/", "695/", "160/", "278/", "330/", "394/", "597/", "598/", "694/", "133/139", "134/139", "135/139", "160/139", "242/139", "243/139", "244/139", "278/139", "298/139", "299/139", "302/139", "303/139", "308/139", "315/139", "330/139", "331/139", "332/139", "333/139", "334/139", "335/139", "336/139", "337/139", "394/139", "395/139", "396/139", "397/139", "398/139", "399/139", "400/139", "401/139", "597/139", "598/139", "694/139", "695/139", "696/139", "697/139", "698/139", "699/139", "700/139", "701/139", "/139", "133/140", "134/140", "135/140", "160/140", "242/140", "243/140", "244/140", "278/140", "298/140", "299/140", "302/140", "303/140", "308/140", "315/140", "330/140", "331/140", "332/140", "333/140", "334/140", "335/140", "336/140", "337/140", "394/140", "395/140", "396/140", "397/140", "398/140", "399/140", "400/140", "401/140", "597/140", "598/140", "694/140", "695/140", "696/140", "697/140", "698/140", "699/140", "700/140", "701/140", "/140", "133/249", "134/249", "135/249", "160/249", "242/249", "243/249", "244/249", "278/249", "298/249", "299/249", "302/249", "303/249", "308/249", "315/249", "330/249", "331/249", "332/249", "333/249", "334/249", "335/249", "336/249", "337/249", "394/249", "395/249", "396/249", "397/249", "398/249", "399/249", "400/249", "401/249", "597/249", "598/249", "694/249", "695/249", "696/249", "697/249", "698/249", "699/249", "700/249", "701/249", "/249", "133/250", "134/250", "135/250", "160/250", "242/250", "243/250", "244/250", "278/250", "298/250", "299/250", "302/250", "303/250", "308/250", "315/250", "330/250", "331/250", "332/250", "333/250", "334/250", "335/250", "336/250", "337/250", "394/250", "395/250", "396/250", "397/250", "398/250", "399/250", "400/250", "401/250", "597/250", "598/250", "694/250", "695/250", "696/250", "697/250", "698/250", "699/250", "700/250", "701/250", "/250", "133/251", "134/251", "135/251", "160/251", "242/251", "243/251", "244/251", "278/251", "298/251", "299/251", "302/251", "303/251", "308/251", "315/251", "330/251", "331/251", "332/251", "333/251", "334/251", "335/251", "336/251", "337/251", "394/251", "395/251", "396/251", "397/251", "398/251", "399/251", "400/251", "401/251", "597/251", "598/251", "694/251", "695/251", "696/251", "697/251", "698/251", "699/251", "700/251", "701/251", "/251", "133/599", "134/599", "135/599", "160/599", "242/599", "243/599", "244/599", "278/599", "298/599", "299/599", "302/599", "303/599", "308/599", "315/599", "330/599", "331/599", "332/599", "333/599", "334/599", "335/599", "336/599", "337/599", "394/599", "395/599", "396/599", "397/599", "398/599", "399/599", "400/599", "401/599", "597/599", "598/599", "694/599", "695/599", "696/599", "697/599", "698/599", "699/599", "700/599", "701/599", "/599", "133/600", "134/600", "135/600", "160/600", "242/600", "243/600", "244/600", "278/600", "298/600", "299/600", "302/600", "303/600", "308/600", "315/600", "330/600", "331/600", "332/600", "333/600", "334/600", "335/600", "336/600", "337/600", "394/600", "395/600", "396/600", "397/600", "398/600", "399/600", "400/600", "401/600", "597/600", "598/600", "694/600", "695/600", "696/600", "697/600", "698/600", "699/600", "700/600", "701/600", "/600", "137/139", "137/140", "137/249", "137/250", "137/251", "137/599", "137/600", "137/", "248/139", "248/140", "248/249", "248/250", "248/251", "248/599", "248/600", "248/", "136/139", "136/140", "136/249", "136/250", "136/251", "136/599", "136/600", "136/", "247/139", "247/140", "247/249", "247/250", "247/251", "247/599", "247/600", "247/", "313/139", "313/140", "313/249", "313/250", "313/251", "313/599", "313/600", "313/", "271/139", "271/140", "271/249", "271/250", "271/251", "271/599", "271/600", "271/"]
                    },
                    l = {
                        variants: {}
                    };

                function d() {
                    return i.storage.local.set({
                        variants: l.variants
                    }).catch((function (e) {
                        console.error("Cannot write variants storage")
                    }))
                }

                function f() {
                    var e = [],
                        t = r.prefs.adaptativeCount,
                        n = 1;
                    for (l.variants.full_list.forEach((function (i) {
                            if (u.test(i)) {
                                if (n > t) return;
                                e.push({
                                    id: "adp:" + n,
                                    label: r._("adaptative", n)
                                }), n++
                            } else {
                                var o = l.variants.full[i];
                                o && e.push({
                                    id: i,
                                    label: o.label,
                                    enabled: o.enabled
                                })
                            }
                        })); n <= t; n++) e.push({
                        id: "adp:" + n,
                        label: r._("adaptative", n)
                    });
                    return e
                }

                function p(e) {
                    l.variants.full_list = [], e.forEach((function (e) {
                        l.variants.full_list.push(e.id), u.test(e.id) || (l.variants.full[e.id].enabled = e.enabled)
                    }))
                }

                function h() {
                    var e = [];
                    return l.variants.adp_list.forEach((function (t) {
                        var n = s.exec(t);
                        if (t) {
                            var i = l.variants.adp_video[n[1]],
                                o = l.variants.adp_audio[n[2]];
                            if (i || o) {
                                var a = [];
                                i && !o ? (a.push(r._("video_only")), a.push(i.extension.toUpperCase())) : !i && o ? (a.push(r._("audio_only")), a.push(o.extension.toUpperCase())) : a.push(i.extension.toUpperCase()), i && (i.size && a.push(i.size), i.fps && a.push(i.fps + " fps"), i.codec && a.push(i.codec)), o && o.codec && a.push(o.codec), e.push({
                                    id: t,
                                    label: a.join(" - ")
                                })
                            }
                        }
                    })), e
                }

                function g(e) {
                    l.variants.adp_list = [], e.forEach((function (e) {
                        l.variants.adp_list.push(e.id)
                    }))
                }
                i.storage.local.get({
                    variants: c
                }).then((function (e) {
                    l.variants = e.variants
                })).catch((function (e) {
                    console.error("Cannot read variants storage")
                }))
            },
            191: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.checkLicense = f, t.validateLicense = p, t.setLicense = h, t.alertAudioNeedsReg = function () {
                    s.alert({
                        title: r._("converter_needs_reg"),
                        text: r._("converter_reg_audio"),
                        buttons: [{
                            text: r._("get_conversion_license"),
                            className: "btn-success",
                            rpcMethod: "goto",
                            rpcArgs: ["https://www.downloadhelper.net/convert" + (u ? "?browser=" + encodeURIComponent(u) : "")]
                        }]
                    })
                }, t.alertHlsDownloadLimit = function () {
                    s.alert({
                        title: r._("chrome_premium_required"),
                        text: r._("chrome_premium_hls", [g]),
                        buttons: [{
                            text: r._("continue"),
                            className: "btn-success",
                            rpcMethod: "goto",
                            rpcArgs: ["https://www.downloadhelper.net/convert" + (u ? "?browser=" + encodeURIComponent(u) : "")]
                        }]
                    })
                };
                var r = n(3),
                    i = r.browser,
                    o = n(34),
                    a = n(7),
                    s = n(194),
                    u = n(8).buildOptions.browser,
                    c = new a.Cache((function () {
                        return i.storage.local.get("license").then((function (e) {
                            return e.license || null
                        }))
                    }), (function (e) {
                        return i.storage.local.set({
                            license: e
                        })
                    })),
                    l = c.get();

                function d(e, t) {
                    var n = new TextEncoder("utf-8").encode(t + e.key + e.email);
                    return crypto.subtle.digest("SHA-256", n).then((function (e) {
                        return a.bufferToHex(e)
                    }))
                }

                function f() {
                    return new Promise((function (e, t) {
                        i.runtime.getPlatformInfo().then((function (n) {
                            if ("linux" == n.os && !r.prefs.linuxLicense) return e({
                                status: "unneeded"
                            });
                            l().then((function (n) {
                                if (null === n) return e({
                                    status: "unset"
                                });
                                var r = {
                                    status: "unset"
                                };
                                if (n.email && (r.email = n.email), n.key && (r.key = n.key), n.name && (r.name = n.name), "mismatch" == n.status) return r.status = "mismatch", r.brLicense = n.brLicense, r.brExt = n.brExt, e(r);
                                o.check().then((function (t) {
                                    return t.status ? d(n, t.info.home) : (r.status = "nocoapp", e(r), null)
                                })).then((function (e) {
                                    if (e) return !n.remoteStatus && r.key ? new Promise((function (t, i) {
                                        p(r.key).then((function (r) {
                                            n = r, t(e)
                                        })).catch((function (e) {
                                            i(e)
                                        }))
                                    })) : e
                                })).then((function (t) {
                                    t && ("accepted" == n.remoteStatus && t === n.sign ? r.status = "accepted" : "blocked" == n.remoteStatus ? r.status = "blocked" : "locked" == n.remoteStatus && (r.status = "locked"), e(r))
                                })).catch(t)
                            })).catch(t)
                        })).catch(t)
                    }))
                }
                async function p(e) {
                    var t = await o.check();
                    if (!t.status) return {
                        key: e,
                        last: Date.now(),
                        status: "nocoapp"
                    };
                    var n = t.info.home,
                        i = void 0;
                    try {
                        i = await a.request({
                            url: "https://www.downloadhelper.net/license-check.json",
                            content: "key=" + encodeURIComponent(e) + "&product=converthelper",
                            headers: {
                                "Content-type": "application/x-www-form-urlencoded"
                            },
                            method: "POST"
                        })
                    } catch (e) {
                        throw new Error(r._("network_error_no_response"))
                    }
                    if (!i.ok) throw new Error(r._("network_error_status", i.status + " " + i.statusText));
                    var s, l = await i.json(),
                        f = {
                            key: e,
                            last: Date.now(),
                            remoteStatus: l.status,
                            status: l.status,
                            name: l.name,
                            email: l.email
                        },
                        p = (s = u) && s.substring(0, 1).toUpperCase() + s.substring(1) || "";
                    if ("fx" != l.target && "firefox" != l.target || "firefox" == u ? "edge" == l.target && "edge" != u ? (f.status = "mismatch", f.brExt = p, f.brLicense = "Edge") : "crx" != l.target && "chrome" != l.target || "chrome" == u || (f.status = "mismatch", f.brExt = p, f.brLicense = "Chrome") : (f.status = "mismatch", f.brExt = p, f.brLicense = "Firefox"), "accepted" == f.status) {
                        var h = await d(f, n);
                        f.sign = h
                    }
                    return await c.set(f), f
                }

                function h(e) {
                    return c.set(e)
                }
                var g = t.hlsDownloadLimit = "edge" === u || "chrome" === u ? 120 : 0;
                r.rpc.listen({
                    checkLicense: f,
                    validateLicense: p,
                    setLicense: h
                })
            },
            49: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.update = s, t.current = function () {
                    return {
                        url: u,
                        urls: c
                    }
                }, t.setTransientTab = p, t.gotoOrOpenTab = function (e) {
                    return d = null, f = null, o.gotoOrOpenTab(e, p)
                };
                var r = n(3).browser,
                    i = n(12),
                    o = n(7),
                    a = null;

                function s() {
                    a && clearTimeout(a), a = setTimeout(l, 50)
                }
                var u = "about:blank",
                    c = {};

                function l() {
                    a = null, r.windows.getCurrent().then((function (e) {
                        return r.tabs.query({
                            windowId: e.id,
                            active: !0
                        })
                    })).then((function (e) {
                        return e.length > 0 ? e[0] : null
                    })).then((function (e) {
                        e && (u = e.url, r.tabs.query({}).then((function (e) {
                            for (var t in c = {}, e) c[e[t].url] = 1;
                            i.dispatch("hits.urlUpdated", {
                                url: u,
                                urls: c
                            })
                        })))
                    }))
                }
                var d = null,
                    f = null;

                function p(e, t) {
                    d = e, f = t
                }
                r.windows.onFocusChanged.addListener(s), r.windows.onRemoved.addListener(s), r.tabs.onActivated.addListener((function (e) {
                    var t = e.tabId;
                    e._windowId, t !== d && (d = null, f = null), s()
                })), r.tabs.onRemoved.addListener((function (e) {
                    d === e && f && r.tabs.update(f, {
                        active: !0
                    }), d = null, f = null, s()
                })), r.tabs.onUpdated.addListener(s), r.tabs.onCreated.addListener((function (e) {
                    "<next-tab>" === d && (d = e.id)
                }))
            },
            7: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                });
                var r = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
                        return typeof e
                    } : function (e) {
                        return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                    },
                    i = function () {
                        function e(e, t) {
                            for (var n = 0; n < t.length; n++) {
                                var r = t[n];
                                r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                            }
                        }
                        return function (t, n, r) {
                            return n && e(t.prototype, n), r && e(t, r), t
                        }
                    }(),
                    o = function (e, t) {
                        if (Array.isArray(e)) return e;
                        if (Symbol.iterator in Object(e)) return function (e, t) {
                            var n = [],
                                r = !0,
                                i = !1,
                                o = void 0;
                            try {
                                for (var a, s = e[Symbol.iterator](); !(r = (a = s.next()).done) && (n.push(a.value), !t || n.length !== t); r = !0);
                            } catch (e) {
                                i = !0, o = e
                            } finally {
                                try {
                                    !r && s.return && s.return()
                                } finally {
                                    if (i) throw o
                                }
                            }
                            return n
                        }(e, t);
                        throw new TypeError("Invalid attempt to destructure non-iterable instance")
                    };

                function a(e, t) {
                    if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                    return !t || "object" != typeof t && "function" != typeof t ? e : t
                }

                function s(e, t) {
                    if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
                    e.prototype = Object.create(t && t.prototype, {
                        constructor: {
                            value: e,
                            enumerable: !1,
                            writable: !0,
                            configurable: !0
                        }
                    }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
                }

                function u(e, t) {
                    if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                }
                t.hash = d, t.hashHex = function (e) {
                    return Math.abs(d(e)).toString(16)
                }, t.gotoTab = f, t.gotoOrOpenTab = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : null,
                        n = 0;

                    function r() {
                        return c.windows.getLastFocused({
                            windowTypes: ["normal"]
                        }).then((function (e) {
                            return "normal" != e.type ? ++n < 20 ? new Promise((function (e, t) {
                                setTimeout((function () {
                                    return r()
                                }), 100)
                            })) : new Promise((function (e, t) {
                                c.windows.getAll({
                                    windowTypes: ["normal"]
                                }).then((function (t) {
                                    if (t.every((function (t) {
                                            return "normal" != t.type || (e(t.id), !1)
                                        }))) throw new Error("No normal window to open tab")
                                }))
                            })) : e.id
                        })).then((function (n) {
                            var r = null;
                            if (n) return c.tabs.query({
                                active: !0,
                                lastFocusedWindow: !0
                            }).then((function (i) {
                                return i.length > 0 && (r = i[0].id), new Promise((function (t, r) {
                                    var i = null,
                                        o = function e(n, r, o) {
                                            n == i && "complete" === r.status && (c.tabs.onUpdated.removeListener(e), t(o))
                                        };
                                    c.tabs.onUpdated.addListener(o), c.tabs.create({
                                        url: e,
                                        windowId: n
                                    }).then((function (e) {
                                        "complete" === e.status ? (c.tabs.onUpdated.removeListener(o), t(e)) : i = e.id
                                    }))
                                })).then((function (e) {
                                    r && t && t(e.id, r)
                                }))
                            }))
                        }))
                    }
                    return f(e).then((function (e) {
                        return e ? Promise.resolve() : r()
                    }))
                }, t.arrayEquals = function (e, t) {
                    if (e.length !== t.length) return !1;
                    for (var n = 0, r = e.length; n < r; n++)
                        if (e[n] !== t[n]) return !1;
                    return !0
                }, t.equals = function (e, t) {
                    return l(e, t)
                }, t.request = v, t.downloadToByteArray = async function (e, t, n) {
                    var r = await v({
                        url: e,
                        headers: t,
                        anonymous: n
                    });
                    if (!r.ok) throw new Error("Request response status " + r.status);
                    var i = await r.arrayBuffer();
                    if (!i) throw new Error("Empty/no response");
                    return new Uint8Array(i)
                }, t.bufferToHex = function (e) {
                    for (var t = [], n = new DataView(e), r = 0; r < n.byteLength; r += 4) {
                        var i = "00000000",
                            o = (i + n.getUint32(r).toString(16)).slice(-8);
                        t.push(o)
                    }
                    return t.join("")
                }, t.Concurrent = function () {
                    for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                    var r = new(Function.prototype.bind.apply(y, [null].concat(t)));
                    return r.callFn().bind(r)
                }, t.isMinimumVersion = function (e, t) {
                    for (var n = e.split(".").map((function (e) {
                            return parseInt(e)
                        })), r = t.split(".").map((function (e) {
                            return parseInt(e)
                        })), i = 0; i < n.length; i++) {
                        if (void 0 === r[i]) return !0;
                        if (n[i] > r[i]) return !0;
                        if (n[i] < r[i]) return !1
                    }
                    return !0
                }, t.executeScriptWithGlobal = async function (e, t, n) {
                    if (i = t, !i || "object" !== (void 0 === i ? "undefined" : r(i))) throw new Error("global argument is not an object");
                    var i;
                    t = function (e) {
                        return JSON.parse(JSON.stringify(e))
                    }(t);
                    var o = {
                        target: e,
                        func: function (e) {
                            Object.assign(window, e)
                        },
                        args: [t]
                    };
                    await c.scripting.executeScript(o), o = {
                        target: e,
                        files: [n]
                    }, await c.scripting.executeScript(o)
                };
                var c = n(3).browser,
                    l = n(64);

                function d(e) {
                    var t, n = 0,
                        r = void 0;
                    if (0 === e.length) return n;
                    for (r = 0, t = e.length; r < t; r++) n = (n << 5) - n + e.charCodeAt(r), n |= 0;
                    return n
                }

                function f(e) {
                    return c.tabs.query({
                        url: e
                    }).then((function (e) {
                        return e.length > 0 && (c.tabs.update(e[0].id, {
                            active: !0
                        }), c.windows.update(e[0].windowId, {
                            focused: !0
                        }), !0)
                    }))
                }
                var p = function () {
                        var e = void 0,
                            t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
                            n = [];
                        for (e = 0; e < 64; e++) n[e] = t[e];
                        var r = [];
                        for (e = 0; e < 64; ++e) r[t.charCodeAt(e)] = e;
                        r["-".charCodeAt(0)] = 62, r["_".charCodeAt(0)] = 63;
                        var i = "undefined" != typeof Uint8Array ? Uint8Array : Array;

                        function o(e) {
                            var t = r[e.charCodeAt(0)];
                            return void 0 !== t ? t : -1
                        }

                        function a(e) {
                            return n[e]
                        }

                        function s(e, t, n) {
                            for (var r, i = void 0, o = [], s = t; s < n; s += 3) i = (e[s] << 16) + (e[s + 1] << 8) + e[s + 2], o.push(a((r = i) >>> 18 >>> 0 & 63) + a(r >>> 12 >>> 0 & 63) + a(r >>> 6 >>> 0 & 63) + a(63 & r));
                            return o.join("")
                        }
                        return {
                            toByteArray: function (e) {
                                var t, n, r = void 0,
                                    a = void 0,
                                    s = void 0,
                                    u = void 0;
                                if (e.length % 4 > 0) throw new Error("Invalid string. Length must be a multiple of 4");
                                var c = e.length;
                                n = "=" === e.charAt(c - 2) ? 2 : "=" === e.charAt(c - 1) ? 1 : 0, u = new i(3 * e.length / 4 - n), t = n > 0 ? e.length - 4 : e.length;
                                var l = 0;

                                function d(e) {
                                    u[l++] = e
                                }
                                for (r = 0, a = 0; r < t; r += 4, a += 3) d((16711680 & (s = o(e.charAt(r)) << 18 | o(e.charAt(r + 1)) << 12 | o(e.charAt(r + 2)) << 6 | o(e.charAt(r + 3)))) >>> 16 >>> 0), d((65280 & s) >>> 8 >>> 0), d((255 & s) >>> 0);
                                return 2 === n ? d(255 & (s = o(e.charAt(r)) << 2 | o(e.charAt(r + 1)) >>> 4 >>> 0)) : 1 === n && (d((s = o(e.charAt(r)) << 10 | o(e.charAt(r + 1)) << 4 | o(e.charAt(r + 2)) >>> 2 >>> 0) >>> 8 >>> 0 & 255), d(255 & s)), u
                            },
                            fromByteArray: function (e) {
                                var t, n = void 0,
                                    r = e.length % 3,
                                    i = "",
                                    o = [],
                                    u = void 0,
                                    c = 16383;
                                for (n = 0, t = e.length - r; n < t; n += c) o.push(s(e, n, n + c > t ? t : n + c));
                                switch (r) {
                                    case 1:
                                        i += a((u = e[e.length - 1]) >>> 2 >>> 0), i += a(u << 4 & 63), i += "==";
                                        break;
                                    case 2:
                                        i += a((u = (e[e.length - 2] << 8) + e[e.length - 1]) >>> 10 >>> 0), i += a(u >>> 4 >>> 0 & 63), i += a(u << 2 & 63), i += "="
                                }
                                return o.push(i), o.join("")
                            }
                        }
                    }(),
                    h = p.toByteArray,
                    g = p.fromByteArray;
                t.toByteArray = h, t.fromByteArray = g;
                var m = ["Accept-Charset", "Accept-Encoding", "Access-Control-Request-Headers", "Access-Control-Request-Method", "Connection", "Content-Length", "Cookie", "Cookie2", "Date", "DNT", "Expect", "Host", "Keep-Alive", "Origin", "Referer", "TE", "Trailer", "Transfer-Encoding", "Upgrade", "Via", "x-chrome-uma-enabled", "x-client-data"];
                async function v(e) {
                    var t = "include";
                    e.anonymous && (t = "omit");
                    var n = e.url,
                        r = e.method || "GET",
                        i = "",
                        a = new Headers;
                    if (e.headers) {
                        if (e.headers instanceof Array) {
                            var s = !0,
                                u = !1,
                                c = void 0;
                            try {
                                for (var l, d = e.headers[Symbol.iterator](); !(s = (l = d.next()).done); s = !0) {
                                    var f = l.value;
                                    a.append(f.name, f.value)
                                }
                            } catch (e) {
                                u = !0, c = e
                            } finally {
                                try {
                                    !s && d.return && d.return()
                                } finally {
                                    if (u) throw c
                                }
                            }
                        } else a = new Headers(e.headers);
                        a.has("referer") && (i = a.get("referer")), a.has("referrer") && (i = a.get("referrer"));
                        var p = [];
                        a.forEach((function (e) {
                            var t = o(e, 2),
                                n = t[0];
                            t[1];
                            (n.startsWith("proxy-") || n.startsWith("sec-")) && p.push(n)
                        }));
                        var h = !0,
                            g = !1,
                            v = void 0;
                        try {
                            for (var y, b = p[Symbol.iterator](); !(h = (y = b.next()).done); h = !0) {
                                var w = y.value;
                                a.delete(w)
                            }
                        } catch (e) {
                            g = !0, v = e
                        } finally {
                            try {
                                !h && b.return && b.return()
                            } finally {
                                if (g) throw v
                            }
                        }
                        var k = !0,
                            x = !1,
                            A = void 0;
                        try {
                            for (var _, O = m[Symbol.iterator](); !(k = (_ = O.next()).done); k = !0) {
                                var I = _.value;
                                a.delete(I)
                            }
                        } catch (e) {
                            x = !0, A = e
                        } finally {
                            try {
                                !k && O.return && O.return()
                            } finally {
                                if (x) throw A
                            }
                        }
                    }
                    var P = void 0;
                    return e.contentJSON ? P = JSON.stringify(e.contentJSON) : e.content && (P = e.content), await fetch(n, {
                        referrer: i,
                        method: r,
                        headers: a,
                        body: P,
                        credentials: t
                    })
                }
                t.Cache = function () {
                    function e(t, n) {
                        u(this, e), this.getFn = t, this.setFn = n, this.callbacks = [], this.queried = !1, this.value = void 0
                    }
                    return i(e, [{
                        key: "get",
                        value: function () {
                            var e = this;
                            return function () {
                                return void 0 !== e.value ? Promise.resolve(e.value) : new Promise((function (t, n) {
                                    if (e.callbacks.push({
                                            resolve: t,
                                            reject: n
                                        }), !e.queried) {
                                        e.queried = !0;
                                        try {
                                            Promise.resolve(e.getFn()).then((function (t) {
                                                for (e.value = t; e.callbacks.length;) e.callbacks.shift().resolve(t)
                                            })).catch((function (t) {
                                                for (; e.callbacks.length;) e.callbacks.shift().reject(t)
                                            }))
                                        } catch (t) {
                                            for (e.queried = !1; e.callbacks.length;) e.callbacks.shift().reject(t)
                                        }
                                    }
                                }))
                            }
                        }
                    }, {
                        key: "set",
                        value: function (e) {
                            if (!this.setFn) return Promise.reject(new Error("Value is read-only"));
                            if (void 0 === e) return Promise.reject(new Error("Cannot set undefined value"));
                            for (this.value = e; this.callbacks.length;) this.callbacks.shift().resolve();
                            return this.setFn(e), Promise.resolve()
                        }
                    }]), e
                }();
                var y = function () {
                    function e() {
                        var t = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : 1;
                        u(this, e), this.maxFn = t, this.pendings = [], this.count = 0
                    }
                    return i(e, [{
                        key: "getMax",
                        value: function () {
                            return Promise.resolve("function" == typeof this.maxFn ? this.maxFn() : this.maxFn)
                        }
                    }, {
                        key: "callFn",
                        value: function () {
                            var e = this;
                            return function (t, n) {
                                return e.getMax().then((function (r) {
                                    return e.count < r ? e.doCall(t) : new Promise((function (r, i) {
                                        var o = function () {
                                            return Promise.resolve(t()).then(r).catch(i)
                                        };
                                        e.pendings.push(o), n && n((function (t) {
                                            var n = e.pendings.indexOf(o);
                                            n >= 0 && (e.pendings.splice(n, 1), r(t))
                                        }), (function (t) {
                                            var n = e.pendings.indexOf(o);
                                            n >= 0 && (e.pendings.splice(n, 1), i(t))
                                        }))
                                    }))
                                }))
                            }
                        }
                    }, {
                        key: "attempt",
                        value: function () {
                            if (this.pendings.length > 0) {
                                var e = this;
                                e.getMax().then((function (t) {
                                    e.count < t && e.doCall(e.pendings.shift())
                                }))
                            }
                        }
                    }, {
                        key: "doCall",
                        value: function (e) {
                            var t = this;
                            return this.count++, Promise.resolve(e()).then((function (e) {
                                return t.count--, t.attempt(), e
                            })).catch((function (e) {
                                throw t.count--, t.attempt(), e
                            }))
                        }
                    }]), e
                }();
                var b = function (e) {
                        function t(e) {
                            u(this, t);
                            var n = a(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                            return n.name = n.constructor.name, "function" == typeof Error.captureStackTrace ? Error.captureStackTrace(n, n.constructor) : n.stack = new Error(e).stack, n
                        }
                        return s(t, Error), t
                    }(),
                    w = t.VDHError = function (e) {
                        function t(e, n) {
                            u(this, t);
                            var r = a(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e));
                            return Object.assign(r, n), r
                        }
                        return s(t, e), t
                    }(b);
                t.DetailsError = function (e) {
                    function t(e, n) {
                        return u(this, t), a(this, (t.__proto__ || Object.getPrototypeOf(t)).call(this, e, {
                            details: n
                        }))
                    }
                    return s(t, e), i(t, [{
                        key: "details",
                        get: function () {
                            return this.details
                        }
                    }]), t
                }(w)
            },
            12: (e, t, n) => {
                "use strict";
                Object.defineProperty(t, "__esModule", {
                    value: !0
                }), t.dispatch = function (e, t) {
                    A.dispatch({
                        type: e,
                        payload: t
                    })
                }, t.getHit = function (e) {
                    return A.getState().hits[e]
                }, t.getHits = function () {
                    return A.getState().hits
                }, t.getLogs = function () {
                    return A.getState().logs
                };
                var r = n(65),
                    i = r.createStore,
                    o = r.combineReducers,
                    a = r.applyMiddleware,
                    s = n(76).createLogger,
                    u = n(216),
                    c = n(3),
                    l = n(1),
                    d = n(50),
                    f = n(189),
                    p = n(190),
                    h = n(192),
                    g = n(191),
                    m = n(244),
                    v = n(224),
                    y = n(222),
                    b = n(79),
                    w = n(197);
                n(49);
                var k = [];
                c.prefs.backgroundReduxLogger && k.push(s({
                    collapsed: function (e, t, n) {
                        return !0
                    }
                }));
                var x = c.browser,
                    A = i(o({
                        hits: d.reducer,
                        progress: d.progressReducer,
                        logs: p.reducer
                    }), a.apply(void 0, k)),
                    _ = u(A.getState, "hits"),
                    O = u(A.getState, "progress"),
                    I = u(A.getState, "logs");

                function P() {
                    var e = A.getState().hits,
                        t = 0,
                        n = 0,
                        r = 0,
                        i = 0;
                    Object.keys(e).forEach((function (o) {
                        switch (e[o].status) {
                            case "running":
                                i++;
                                break;
                            case "active":
                                t++, n++;
                                break;
                            case "inactive":
                                n++;
                                break;
                            case "pinned":
                                r++
                        }
                    }));
                    var o = !1;
                    (0 == n || "currenttab" == c.prefs.iconActivation && 0 == t) && (o = !0), x.action.setIcon({
                        path: {
                            32: "/content/images/icon-32" + (o ? "-off" : "") + ".png",
                            40: "/content/images/icon-40" + (o ? "-off" : "") + ".png",
                            48: "/content/images/icon-48" + (o ? "-off" : "") + ".png",
                            128: "/content/images/icon-128" + (o ? "-off" : "") + ".png"
                        }
                    });
                    var a = "",
                        s = "#000";
                    switch (c.prefs.iconBadge) {
                        case "tasks":
                            s = "#00f", a = i || "";
                            break;
                        case "activetab":
                            s = "#080", a = t || "";
                            break;
                        case "anytab":
                            s = "#b59e32", a = n || "";
                            break;
                        case "pinned":
                            s = "#000", a = r || "";
                            break;
                        case "mixed":
                            r > 0 ? (s = "#000", a = r) : i > 0 ? (s = "#00f", a = i) : t > 0 ? (s = "#080", a = t) : n > 0 && (s = "#b59e32", a = n)
                    }
                    var u = A.getState().logs.filter((function (e) {
                        return "error" === e.type
                    }));
                    u.length > 0 && (a = u.length, s = "#f44"), x.action.setBadgeText({
                        text: "" + a
                    }), x.action.setBadgeBackgroundColor({
                        color: s
                    })
                }
                A.subscribe(_((function () {
                    var e = A.getState().hits;
                    l.call("main", "hits", e);
                    try {
                        m.updateHits(e)
                    } catch (e) {
                        console.error(e)
                    }
                    P()
                }))), A.subscribe(O((function () {
                    l.call("main", "progress", A.getState().progress)
                }))), A.subscribe(I((function () {
                    l.call("main", "logs", A.getState().logs), P()
                }))), l.listen({
                    getHits: function () {
                        return A.getState().hits
                    },
                    getHit: function (e) {
                        return A.getState().hits[e]
                    },
                    getMainData: function () {
                        return {
                            hits: A.getState().hits,
                            actions: f.describeAll(),
                            logs: A.getState().logs,
                            progress: A.getState().progress
                        }
                    },
                    hitPageData: function (e) {
                        d.updateOriginal(e.id, e.data)
                    },
                    closePopup: function () {
                        return l.call("main", "close")
                    },
                    closePanel: function (e) {
                        return c.ui.close(e)
                    }
                }), P(), c.rpc.listen({
                    exportSettings: function () {
                        return x.storage.local.get(["blacklist", "license", "variants", "convrules", "outputConfigs", "smartname"]).then((function (e) {
                            var t = Object.assign({
                                    blacklist: {},
                                    license: null,
                                    conversionRules: [],
                                    outputConfigs: {}
                                }, e, {
                                    "weh-prefs": c.prefs.getAll()
                                }),
                                n = "data:," + JSON.stringify(t, null, 4);
                            x.downloads.download({
                                url: n,
                                filename: "vdh-settings.json",
                                saveAs: !0,
                                conflictAction: "uniquify"
                            })
                        }))
                    },
                    importSettings: function (e) {
                        return new Promise((function (t, n) {
                            e.convrules && w.set(e.convrules), e.outputConfigs && h.setOutputConfigs(e.outputConfigs), e.license && g.setLicense(e.license), e.blacklist && v.set(e.blacklist), e.variants && y.setVariants(e.variants), e.smartname && b.set(e.smartname), t(e["weh-prefs"] || {})
                        }))
                    },
                    reloadAddon: function () {
                        x.runtime.reload()
                    }
                })
            },
            240: (e, t, n) => {
                "use strict";
                t.S = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : 6e4;
                    return async function () {
                        if (a.has(e)) return a.get(e).apply(void 0, arguments);
                        var n = await async function (e, t) {
                            i || (s || (s = chrome.offscreen.createDocument({
                                url: "content/workerfactory.html",
                                reasons: [chrome.offscreen.Reason.WORKERS],
                                justification: "Working around the lack of worker in Service Worker"
                            })), await s);
                            var n = new BroadcastChannel(e),
                                u = new Promise((function (t, r) {
                                    var i = function e(r) {
                                        "worker-ready" == r.data && (n.removeEventListener("message", e), t())
                                    };
                                    n.addEventListener("message", i), o.postMessage({
                                        type: "spawn-worker",
                                        path: e
                                    })
                                }));
                            await u;
                            var c = function (e, t) {
                                    n.postMessage(t)
                                },
                                l = function (t) {
                                    "weh#rpc" == t.data.type && r.receive(t.data, c, e)
                                };
                            n.addEventListener("message", l);
                            var d = function () {
                                    var t = function t(r) {
                                        var u = "worker-killed" == r.data.type,
                                            c = r.data.worker_path == e;
                                        u && c && (a.delete(e), n.close(), o.removeEventListener("message", t), i || 0 != a.size || (s = null, chrome.offscreen.closeDocument()))
                                    };
                                    o.addEventListener("message", t), o.postMessage({
                                        type: "kill-worker",
                                        path: e
                                    })
                                },
                                f = void 0;

                            function p() {
                                f = setTimeout(d, t)
                            }

                            function h() {
                                clearTimeout(f)
                            }
                            var g = 0;
                            return p(e),
                                function () {
                                    for (var t = arguments.length, n = Array(t), i = 0; i < t; i++) n[i] = arguments[i];
                                    g++, h();
                                    try {
                                        return r.call.apply(r, [c, e].concat(n))
                                    } finally {
                                        0 == --g && p(e)
                                    }
                                }
                        }(e, t);
                        return a.set(e, n), n.apply(void 0, arguments)
                    }
                };
                var r = n(1),
                    i = "firefox" == n(8).buildOptions.browser,
                    o = new BroadcastChannel("workerfactory-inner"),
                    a = new Map,
                    s = null;
                i && n(195)
            },
            64: (e, t, n) => {
                var r = Array.prototype.slice,
                    i = n(74),
                    o = n(75),
                    a = e.exports = function (e, t, n) {
                        return n || (n = {}), e === t || (e instanceof Date && t instanceof Date ? e.getTime() === t.getTime() : !e || !t || "object" != typeof e && "object" != typeof t ? n.strict ? e === t : e == t : function (e, t, n) {
                            var c, l;
                            if (s(e) || s(t)) return !1;
                            if (e.prototype !== t.prototype) return !1;
                            if (o(e)) return !!o(t) && (e = r.call(e), t = r.call(t), a(e, t, n));
                            if (u(e)) {
                                if (!u(t)) return !1;
                                if (e.length !== t.length) return !1;
                                for (c = 0; c < e.length; c++)
                                    if (e[c] !== t[c]) return !1;
                                return !0
                            }
                            try {
                                var d = i(e),
                                    f = i(t)
                            } catch (e) {
                                return !1
                            }
                            if (d.length != f.length) return !1;
                            for (d.sort(), f.sort(), c = d.length - 1; c >= 0; c--)
                                if (d[c] != f[c]) return !1;
                            for (c = d.length - 1; c >= 0; c--)
                                if (l = d[c], !a(e[l], t[l], n)) return !1;
                            return typeof e == typeof t
                        }(e, t, n))
                    };

                function s(e) {
                    return null == e
                }

                function u(e) {
                    return !(!e || "object" != typeof e || "number" != typeof e.length) && ("function" == typeof e.copy && "function" == typeof e.slice && !(e.length > 0 && "number" != typeof e[0]))
                }
            },
            75: (e, t) => {
                var n = "[object Arguments]" == function () {
                    return Object.prototype.toString.call(arguments)
                }();

                function r(e) {
                    return "[object Arguments]" == Object.prototype.toString.call(e)
                }

                function i(e) {
                    return e && "object" == typeof e && "number" == typeof e.length && Object.prototype.hasOwnProperty.call(e, "callee") && !Object.prototype.propertyIsEnumerable.call(e, "callee") || !1
                }(t = e.exports = n ? r : i).supported = r, t.unsupported = i
            },
            74: (e, t) => {
                function n(e) {
                    var t = [];
                    for (var n in e) t.push(n);
                    return t
                }(e.exports = "function" == typeof Object.keys ? Object.keys : n).shim = n
            },
            210: function (e, t) {
                var n, r, i;
                ! function (o, a) {
                    "use strict";
                    "object" == typeof e.exports ? e.exports = a() : (r = [], void 0 === (i = "function" == typeof (n = a) ? n.apply(t, r) : n) || (e.exports = i))
                }(0, (function () {
                    "use strict";
                    var e = Object.prototype.toString;

                    function t(e, t) {
                        return null != e && Object.prototype.hasOwnProperty.call(e, t)
                    }

                    function n(e) {
                        if (!e) return !0;
                        if (i(e) && 0 === e.length) return !0;
                        if ("string" != typeof e) {
                            for (var n in e)
                                if (t(e, n)) return !1;
                            return !0
                        }
                        return !1
                    }

                    function r(t) {
                        return e.call(t)
                    }
                    var i = Array.isArray || function (t) {
                        return "[object Array]" === e.call(t)
                    };

                    function o(e) {
                        var t = parseInt(e);
                        return t.toString() === e ? t : e
                    }

                    function a(e) {
                        var a, s = function (e) {
                            return Object.keys(s).reduce((function (t, n) {
                                return "create" === n || "function" == typeof s[n] && (t[n] = s[n].bind(s, e)), t
                            }), {})
                        };

                        function u(e, t) {
                            if (a(e, t)) return e[t]
                        }

                        function c(t, n, r, i) {
                            if ("number" == typeof n && (n = [n]), !n || 0 === n.length) return t;
                            if ("string" == typeof n) return c(t, n.split(".").map(o), r, i);
                            var a = n[0],
                                s = u(t, a);
                            if (e.includeInheritedProps && ("__proto__" === a || "constructor" === a && "function" == typeof s)) throw new Error("For security reasons, object's magic properties cannot be set");
                            return 1 === n.length ? (void 0 !== s && i || (t[a] = r), s) : (void 0 === s && ("number" == typeof n[1] ? t[a] = [] : t[a] = {}), c(t[a], n.slice(1), r, i))
                        }
                        return a = (e = e || {}).includeInheritedProps ? function () {
                            return !0
                        } : function (e, n) {
                            return "number" == typeof n && Array.isArray(e) || t(e, n)
                        }, s.has = function (n, r) {
                            if ("number" == typeof r ? r = [r] : "string" == typeof r && (r = r.split(".")), !r || 0 === r.length) return !!n;
                            for (var a = 0; a < r.length; a++) {
                                var s = o(r[a]);
                                if (!("number" == typeof s && i(n) && s < n.length || (e.includeInheritedProps ? s in Object(n) : t(n, s)))) return !1;
                                n = n[s]
                            }
                            return !0
                        }, s.ensureExists = function (e, t, n) {
                            return c(e, t, n, !0)
                        }, s.set = function (e, t, n, r) {
                            return c(e, t, n, r)
                        }, s.insert = function (e, t, n, r) {
                            var o = s.get(e, t);
                            r = ~~r, i(o) || (o = [], s.set(e, t, o)), o.splice(r, 0, n)
                        }, s.empty = function (e, t) {
                            var o, u;
                            if (!n(t) && (null != e && (o = s.get(e, t)))) {
                                if ("string" == typeof o) return s.set(e, t, "");
                                if (function (e) {
                                        return "boolean" == typeof e || "[object Boolean]" === r(e)
                                    }(o)) return s.set(e, t, !1);
                                if ("number" == typeof o) return s.set(e, t, 0);
                                if (i(o)) o.length = 0;
                                else {
                                    if (! function (e) {
                                            return "object" == typeof e && "[object Object]" === r(e)
                                        }(o)) return s.set(e, t, null);
                                    for (u in o) a(o, u) && delete o[u]
                                }
                            }
                        }, s.push = function (e, t) {
                            var n = s.get(e, t);
                            i(n) || (n = [], s.set(e, t, n)), n.push.apply(n, Array.prototype.slice.call(arguments, 2))
                        }, s.coalesce = function (e, t, n) {
                            for (var r, i = 0, o = t.length; i < o; i++)
                                if (void 0 !== (r = s.get(e, t[i]))) return r;
                            return n
                        }, s.get = function (e, t, n) {
                            if ("number" == typeof t && (t = [t]), !t || 0 === t.length) return e;
                            if (null == e) return n;
                            if ("string" == typeof t) return s.get(e, t.split("."), n);
                            var r = o(t[0]),
                                i = u(e, r);
                            return void 0 === i ? n : 1 === t.length ? i : s.get(e[r], t.slice(1), n)
                        }, s.del = function (e, t) {
                            if ("number" == typeof t && (t = [t]), null == e) return e;
                            if (n(t)) return e;
                            if ("string" == typeof t) return s.del(e, t.split("."));
                            var r = o(t[0]);
                            return a(e, r) ? 1 !== t.length ? s.del(e[r], t.slice(1)) : (i(e) ? e.splice(r, 1) : delete e[r], e) : e
                        }, s
                    }
                    var s = a();
                    return s.create = a, s.withInheritedProps = a({
                        includeInheritedProps: !0
                    }), s
                }))
            },
            216: (e, t, n) => {
                "use strict";
                var r = n(210).get;

                function i(e, t) {
                    return e === t
                }
                e.exports = function (e, t, n) {
                    n = n || i;
                    var o = r(e(), t);
                    return function (i) {
                        return function () {
                            var a = r(e(), t);
                            if (!n(o, a)) {
                                var s = o;
                                o = a, i(a, s, t)
                            }
                        }
                    }
                }
            },
            3: (e, t, n) => {
                "use strict";
                var r = n(0),
                    i = r.browser,
                    o = {};
                r.rpc = n(1), r.rpc.setUseTarget(!0), r.rpc.setPost((function (e, t) {
                    var n = o[e];
                    n && n.port && n.port.postMessage(t)
                })), r.rpc.listen({
                    appStarted: function (e) {},
                    appReady: function (e) {},
                    closePanel: function (e) {
                        r.ui.close(e)
                    }
                }), i.runtime.onConnect.addListener((function (e) {
                    /^weh:(.*?):(.*)/.exec(e.name) && (e.onMessage.addListener((function (t) {
                        if (void 0 !== t._method && ("appStarted" === t._method || "appReady" === t._method)) {
                            var n = t._args[0] && t._args[0].uiName || null,
                                i = o[n] || {
                                    ready: !1
                                };
                            if (o[n] = i, Object.assign(i, t._args[0], {
                                    port: e
                                }), "appReady" == t._method) {
                                i.ready = !0, i.initData && setTimeout((function () {
                                    r.rpc.call(n, "wehInitData", i.initData)
                                }), 0);
                                var a = p[n];
                                a && a.timer && (clearTimeout(a.timer), delete a.timer)
                            }
                            e._weh_app = n
                        }
                        r.rpc.receive(t, e.postMessage.bind(e), e._weh_app)
                    })), e.onDisconnect.addListener((function () {
                        var t = e._weh_app;
                        if (t) {
                            delete o[t];
                            var n = p[t];
                            n && (n.timer && clearTimeout(n.timer), delete p[t], n.reject(new Error("Disconnected waiting for " + t)))
                        }
                    })))
                })), r.__declareAppTab = function (e, t) {
                    o[e] || (o[e] = {}), Object.assign(o[e], t)
                }, r.__closeByTab = function (e) {
                    Object.keys(o).forEach((function (t) {
                        if (o[t].tab === e) {
                            delete o[t];
                            var n = p[t];
                            n && (n.timer && clearTimeout(n.timer), delete p[t], n.reject(new Error("Disconnected waiting for " + t)))
                        }
                    }))
                }, r._ = n(11).getMessage, r.ui = n(212), r.openedContents = function () {
                    return Object.keys(o)
                };
                var a = n(33);

                function s(e) {
                    var t = 0;
                    if (0 === e.length) return t;
                    for (var n = 0; n < e.length; n++) t = (t << 5) - t + e.charCodeAt(n), t &= t;
                    return t
                }
                r.prefs = a;
                var u, c, l = 0,
                    d = {};
                try {
                    var f = localStorage.getItem("weh-prefs");
                    null === f ? i.storage.local.get("weh-prefs").then((function (e) {
                        var t = e["weh-prefs"];
                        t && a.assign(t)
                    })) : (u = f, c = {}, JSON.parse(u).forEach((function (e) {
                        c[e.name] = e.value
                    })), d = c, l = s(f))
                } catch (e) {}
                a.assign(d), a.on("", {
                    pack: !0
                }, (function (e, t) {
                    Object.assign(d, e);
                    var n = function (e) {
                            return JSON.stringify(Object.keys(e).sort().map((function (t) {
                                return {
                                    name: t,
                                    value: e[t]
                                }
                            })))
                        }(d),
                        a = s(n);
                    a != l && (l = a, localStorage.setItem("weh-prefs", n), i.storage.local.set({
                        "weh-prefs": d
                    })), Object.keys(o).forEach((function (t) {
                        o[t].usePrefs && r.rpc.call(t, "setPrefs", e)
                    }))
                }));
                var p = {};
                r.wait = function (e) {
                    var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {},
                        n = p[e];
                    return n && (n.timer && clearTimeout(n.timer), delete p[e], n.reject(new Error("Waiter for " + e + " overriden"))), new Promise((function (n, r) {
                        p[e] = {
                            resolve: n,
                            reject: r,
                            timer: setTimeout((function () {
                                delete p[e], r(new Error("Waiter for " + e + " timed out"))
                            }), t.timeout || 6e4)
                        }
                    }))
                }, r.rpc.listen({
                    prefsGetAll: function () {
                        return a.getAll()
                    },
                    prefsGetSpecs: function () {
                        return a.getSpecs()
                    },
                    prefsSet: function (e) {
                        return a.assign(e)
                    },
                    trigger: function (e, t) {
                        var n = p[e];
                        if (!n) throw new Error("No waiter for", e);
                        n.timer && (clearTimeout(n.timer), delete n.timer), delete p[e], n.resolve(t)
                    }
                }), e.exports = r
            },
            8: e => {
                e.exports = {
                    buildDate: "Tue Aug 22 2023 18:17:53 GMT+0200 (Central European Summer Time)",
                    buildOptions: {
                        browser: "firefox"
                    },
                    prod: !0
                }
            },
            11: (e, t, n) => {
                "use strict";
                var r = n(0).browser,
                    i = {},
                    o = new RegExp("\\$[a-zA-Z]*([0-9]+)\\$", "g");

                function a() {
                    try {
                        null === (i = JSON.parse(window.localStorage.getItem("wehI18nCustom"))) && (i = {}, r.storage.local.get("wehI18nCustom").then((function (e) {
                            var t = e.wehI18nCustom;
                            t && Object.assign(i, t)
                        })))
                    } catch (e) {
                        i = {}
                    }
                }
                a(), e.exports = {
                    getMessage: function (e, t) {
                        if (/-/.test(e)) {
                            var n = e.replace(/-/g, "_");
                            console.warn("Wrong i18n message name. Should it be", n, "instead of", e, "?"), e = n
                        }
                        var a = i[e];
                        return a && a.message.length > 0 ? (Array.isArray(t) || (t = [t]), (a.message || "").replace(o, (function (e) {
                            var n = o.exec(e);
                            return n && t[parseInt(n[1]) - 1] || "??"
                        }))) : r.i18n.getMessage.apply(r.i18n, arguments)
                    },
                    reload: a
                }
            },
            214: (e, t, n) => {
                "use strict";
                var r = n(0),
                    i = n(1),
                    o = n(33),
                    a = r.browser,
                    s = null,
                    u = null,
                    c = !1;
                a.runtime.onMessageExternal && (a.runtime.onMessageExternal.addListener((function (e, t, n) {
                    switch (e.type) {
                        case "weh#inspect-ping":
                            s = t.id, n({
                                type: "weh#inspect-pong",
                                version: 1,
                                manifest: a.runtime.getManifest()
                            });
                            break;
                        case "weh#inspect":
                            s = t.id, (c = e.inspected) ? i.setHook((function (e) {
                                c && s && a.runtime.sendMessage(s, {
                                    type: "weh#inspect-message",
                                    message: e
                                }).catch((function (e) {
                                    console.info("Error sending message", e), c = !1
                                }))
                            })) : i.setHook(null), n({
                                type: "weh#inspect",
                                version: 1,
                                inspected: c
                            });
                            break;
                        case "weh#get-prefs":
                            s = t.id, n({
                                type: "weh#prefs",
                                prefs: o.getAll(),
                                specs: o.getSpecs()
                            });
                            break;
                        case "weh#set-pref":
                            o[e.pref] = e.value, n(!0);
                            break;
                        case "weh#get-storage":
                            s = t.id, new Promise((function (e, t) {
                                var n = {};
                                ["localStorage", "sessionStorage"].forEach((function (e) {
                                    try {
                                        var t = window[e];
                                        if (t) {
                                            for (var r = {}, i = 0; i < t.length; i++) {
                                                var o = t.key(i),
                                                    a = t.getItem(o);
                                                try {
                                                    r[o] = JSON.parse(a)
                                                } catch (e) {
                                                    r[o] = a
                                                }
                                            }
                                            n[e] = r
                                        }
                                    } catch (e) {}
                                }));
                                var r = [];
                                ["local", "sync", "managed"].forEach((function (e) {
                                    try {
                                        var t = a.storage && a.storage[e];
                                        if (t) return new Promise((function (i, o) {
                                            var a = t.get(null).then((function (t) {
                                                n[e] = t
                                            })).catch((function (e) {}));
                                            r.push(a)
                                        }))
                                    } catch (e) {}
                                })), Promise.all(r).then((function () {
                                    e(n)
                                })).catch(t)
                            })).then((function (e) {
                                a.runtime.sendMessage(s, {
                                    type: "weh#storage",
                                    storage: e
                                })
                            })).catch((function (e) {
                                console.error("Error get storage data", e)
                            })), n({
                                type: "weh#storage-pending"
                            })
                    }
                })), u = {
                    send: function () {
                        console.info("TODO implement inspect.send")
                    }
                }), e.exports = u
            },
            215: (e, t, n) => {
                "use strict";
                var r = function () {
                    function e(e, t) {
                        for (var n = 0; n < t.length; n++) {
                            var r = t[n];
                            r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                        }
                    }
                    return function (t, n, r) {
                        return n && e(t.prototype, n), r && e(t, r), t
                    }
                }();

                function i(e) {
                    if (Array.isArray(e)) {
                        for (var t = 0, n = Array(e.length); t < e.length; t++) n[t] = e[t];
                        return n
                    }
                    return Array.from(e)
                }

                function o(e, t) {
                    if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                }
                var a = n(0).browser,
                    s = n(1),
                    u = function () {
                        function e() {
                            o(this, e), this.listeners = []
                        }
                        return r(e, [{
                            key: "addListener",
                            value: function (e) {
                                this.listeners.push(e)
                            }
                        }, {
                            key: "removeListener",
                            value: function (e) {
                                this.listeners = this.listeners.filter((function (t) {
                                    return e !== t
                                }))
                            }
                        }, {
                            key: "removeAllListeners",
                            value: function () {
                                this.listeners = []
                            }
                        }, {
                            key: "notify",
                            value: function () {
                                for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                                this.listeners.forEach((function (e) {
                                    try {
                                        e.apply(void 0, t)
                                    } catch (e) {
                                        console.warn(e)
                                    }
                                }))
                            }
                        }]), e
                    }(),
                    c = function () {
                        function e(t) {
                            var n = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                            o(this, e), this.appId = t, this.name = n.name || t, this.appPort = null, this.pendingCalls = [], this.runningCalls = [], this.state = "idle", this.postFn = this.post.bind(this), this.postMessageFn = this.postMessage.bind(this), this.onAppNotFound = new u, this.onAppNotFoundCheck = new u, this.onCallCount = new u, this.appStatus = "unknown", this.app2AddonCallCount = 0, this.addon2AppCallCount = 0
                        }
                        return r(e, [{
                            key: "post",
                            value: function (e, t) {
                                this.appPort.postMessage(t)
                            }
                        }, {
                            key: "postMessage",
                            value: function (e) {
                                this.appPort.postMessage(e)
                            }
                        }, {
                            key: "updateCallCount",
                            value: function (e, t) {
                                switch (e) {
                                    case 2:
                                        this.app2AddonCallCount += t;
                                        break;
                                    case 1:
                                        this.addon2AppCallCount += t
                                }
                                this.onCallCount.notify(this.addon2AppCallCount, this.app2AddonCallCount)
                            }
                        }, {
                            key: "close",
                            value: function () {
                                if (this.appPort) try {
                                    this.appPort.disconnect(), this.cleanup()
                                } catch (e) {}
                            }
                        }, {
                            key: "call",
                            value: function () {
                                for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                                return this.callCatchAppNotFound.apply(this, [null].concat(t))
                            }
                        }, {
                            key: "callCatchAppNotFound",
                            value: function (e) {
                                for (var t = arguments.length, n = Array(t > 1 ? t - 1 : 0), r = 1; r < t; r++) n[r - 1] = arguments[r];
                                var o = this;

                                function u(e) {
                                    for (var t; t = o.pendingCalls.shift();) e ? t.reject(e) : function () {
                                        o.runningCalls.push(t);
                                        var e = t;
                                        s.call.apply(s, [o.postFn, o.name].concat(i(t.params))).then((function (t) {
                                            return o.runningCalls.splice(o.runningCalls.indexOf(e), 1), t
                                        })).then(e.resolve).catch((function (t) {
                                            o.runningCalls.splice(o.runningCalls.indexOf(e), 1), e.reject(t)
                                        }))
                                    }()
                                }
                                switch (!e || "unknown" != o.appStatus && "checking" != o.appStatus || o.onAppNotFoundCheck.addListener(e), o.updateCallCount(1, 1), this.state) {
                                    case "running":
                                        return new Promise((function (e, t) {
                                            var r = {
                                                resolve: e,
                                                reject: t,
                                                params: [].concat(n)
                                            };
                                            o.runningCalls.push(r), s.call.apply(s, [o.postFn, o.name].concat(n)).then((function (e) {
                                                return o.runningCalls.splice(o.runningCalls.indexOf(r), 1), e
                                            })).then(r.resolve).catch((function (e) {
                                                o.runningCalls.splice(o.runningCalls.indexOf(r), 1), r.reject(e)
                                            }))
                                        })).then((function (e) {
                                            return o.updateCallCount(1, -1), e
                                        })).catch((function (e) {
                                            throw o.updateCallCount(1, -1), e
                                        }));
                                    case "idle":
                                        return o.state = "pending", new Promise((function (t, r) {
                                            o.pendingCalls.push({
                                                resolve: t,
                                                reject: r,
                                                params: [].concat(n)
                                            });
                                            var i = a.runtime.connectNative(o.appId);
                                            o.appStatus = "checking", o.appPort = i, i.onMessage.addListener((function (e) {
                                                "checking" == o.appStatus && (o.appStatus = "ok", o.onAppNotFoundCheck.removeAllListeners()), s.receive(e, o.postMessageFn, o.name)
                                            })), i.onDisconnect.addListener((function () {
                                                u(new Error("Disconnected")), o.cleanup(), "checking" != o.appStatus || e || o.onAppNotFound.notify(o.appPort && o.appPort.error || a.runtime.lastError)
                                            })), o.state = "running", u()
                                        })).then((function (e) {
                                            return o.updateCallCount(1, -1), e
                                        })).catch((function (e) {
                                            throw o.updateCallCount(1, -1), e
                                        }));
                                    case "pending":
                                        return new Promise((function (e, t) {
                                            o.pendingCalls.push({
                                                resolve: e,
                                                reject: t,
                                                params: [].concat(n)
                                            })
                                        })).then((function (e) {
                                            return o.updateCallCount(1, -1), e
                                        })).catch((function (e) {
                                            throw o.updateCallCount(1, -1), e
                                        }))
                                }
                            }
                        }, {
                            key: "listen",
                            value: function (e) {
                                var t = this,
                                    n = {};
                                return Object.keys(e).forEach((function (r) {
                                    n[r] = function () {
                                        return t.updateCallCount(2, 1), Promise.resolve(e[r].apply(e, arguments)).then((function (e) {
                                            return t.updateCallCount(2, -1), e
                                        })).catch((function (e) {
                                            throw t.updateCallCount(2, -1), e
                                        }))
                                    }
                                })), s.listen(n)
                            }
                        }, {
                            key: "cleanup",
                            value: function () {
                                var e, t = this;
                                for ("checking" == t.appStatus && (t.onAppNotFoundCheck.notify(t.appPort && t.appPort.error || a.runtime.lastError), t.onAppNotFoundCheck.removeAllListeners()); e = t.runningCalls.shift();) e.reject(new Error("Native port disconnected"));
                                t.state = "idle", t.appStatus = "unknown", t.appPort = null
                            }
                        }]), e
                    }();
                e.exports = function () {
                    for (var e = arguments.length, t = Array(e), n = 0; n < e; n++) t[n] = arguments[n];
                    return new(Function.prototype.bind.apply(c, [null].concat(t)))
                }
            },
            33: (e, t, n) => {
                "use strict";
                var r = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
                        return typeof e
                    } : function (e) {
                        return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
                    },
                    i = n(11).getMessage;

                function o() {
                    this.$specs = {}, this.$values = null, this.$values || (this.$values = {}), this.$listeners = {}
                }
                o.prototype = {
                    notify: function (e, t, n, r) {
                        for (var i = this, o = e.split("."), a = [], s = o.length; s >= 0; s--) a.push(o.slice(0, s).join("."));
                        a.forEach((function (o) {
                            var a = i.$listeners[o];
                            a && a.forEach((function (i) {
                                if (i.specs == r)
                                    if (i.pack) i.pack[e] = t, void 0 === i.old[e] && (i.old[e] = n), i.timer && clearTimeout(i.timer), i.timer = setTimeout((function () {
                                        delete i.timer;
                                        var e = i.pack,
                                            t = i.old;
                                        i.pack = {}, i.old = {};
                                        try {
                                            i.callback(e, t)
                                        } catch (e) {}
                                    }), 0);
                                    else try {
                                        i.callback(e, t, n)
                                    } catch (e) {}
                            }))
                        }))
                    },
                    forceNotify: function (e) {
                        void 0 === e && (e = !1);
                        var t = this;
                        Object.keys(t.$specs).forEach((function (n) {
                            t.notify(n, t.$values[n], t.$values[n], e)
                        }))
                    },
                    declare: function (e) {
                        var t = this;
                        Array.isArray(e) || (e = Object.keys(e).map((function (t) {
                            var n = e[t];
                            return n.name = t, n
                        }))), e.forEach((function (e) {
                            if (s[e.name]) throw new Error("Forbidden prefs key " + e.name);
                            if (e.hidden) e.label = e.name, e.description = "";
                            else {
                                var n = e.name.replace(/[^0-9a-zA-Z_]/g, "_");
                                e.label = e.label || i("weh_prefs_label_" + n) || e.name, e.description = e.description || i("weh_prefs_description_" + n) || ""
                            }
                            "choice" == e.type && (e.choices = (e.choices || []).map((function (t) {
                                if ("object" == (void 0 === t ? "undefined" : r(t))) return t;
                                if (e.hidden) return {
                                    value: t,
                                    name: t
                                };
                                var o = t.replace(/[^0-9a-zA-Z_]/g, "_");
                                return {
                                    value: t,
                                    name: i("weh_prefs_" + n + "_option_" + o) || t
                                }
                            })));
                            var o, a = null;
                            t.$specs[e.name] || (o = e.name, void 0 !== t[e.name] && (a = t[e.name]), Object.defineProperty(t, o, {
                                set: function (e) {
                                    var n = t.$values[o];
                                    n !== e && (t.$values[o] = e, t.notify(o, e, n, !1))
                                },
                                get: function () {
                                    return void 0 !== t.$values[o] ? t.$values[o] : t.$specs[o] && t.$specs[o].defaultValue || void 0
                                }
                            }));
                            var u = t.$specs[e.name];
                            t.$specs[e.name] = e, null !== a ? t.$values[e.name] = a : void 0 === t.$values[e.name] && (t.$values[e.name] = e.defaultValue), t.notify(e.name, e, u, !0)
                        }))
                    },
                    on: function () {
                        var e = "",
                            t = {},
                            n = 0;
                        "string" == typeof arguments[n] && (e = arguments[n++]), "object" == r(arguments[n]) && (t = arguments[n++]);
                        var i = arguments[n],
                            o = !!t.pack;
                        this.$listeners[e] || (this.$listeners[e] = []);
                        var a = {
                            callback: i,
                            specs: !!t.specs
                        };
                        o && (a.pack = {}, a.old = {}), this.$listeners[e].push(a)
                    },
                    off: function () {
                        var e = "",
                            t = 0;
                        "string" == typeof arguments[t] && (e = arguments[t++]);
                        var n = arguments[t],
                            r = this.$listeners[e];
                        if (r)
                            for (var i = r.length - 1; i >= 0; i--) n && r[i] != n || r.splice(i, 1)
                    },
                    getAll: function () {
                        return Object.assign({}, this.$values)
                    },
                    getSpecs: function () {
                        return Object.assign({}, this.$specs)
                    },
                    assign: function (e) {
                        for (var t in e) e.hasOwnProperty(t) && (this[t] = e[t])
                    },
                    isValid: function (e, t) {
                        var n = this.$specs[e];
                        if (n) {
                            switch (n.type) {
                                case "string":
                                    if (n.regexp && !new RegExp(n.regexp).test(t)) return !1;
                                    break;
                                case "integer":
                                    if (!/^-?[0-9]+$/.test(t)) return !1;
                                    if (isNaN(parseInt(t))) return !1;
                                case "float":
                                    if ("float" == n.type) {
                                        if (!/^-?[0-9]+(\.[0-9]+)?|(\.[0-9]+)$/.test(t)) return !1;
                                        if (isNaN(parseFloat(t))) return !1
                                    }
                                    if (void 0 !== n.minimum && t < n.minimum) return !1;
                                    if (void 0 !== n.maximum && t > n.maximum) return !1;
                                    break;
                                case "choice":
                                    var r = !1;
                                    if ((n.choices || []).forEach((function (e) {
                                            t == e.value && (r = !0)
                                        })), !r) return !1
                            }
                            return !0
                        }
                    },
                    reducer: function () {
                        var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                            t = arguments[1];
                        if ("weh.SET_PREFS" === t.type) e = Object.assign({}, e, t.payload);
                        return e
                    },
                    reduxDispatch: function (e) {
                        this.on("", {
                            pack: !0
                        }, (function (t) {
                            e.dispatch({
                                type: "weh.SET_PREFS",
                                payload: t
                            })
                        }))
                    }
                };
                var a = new o,
                    s = {};
                for (var u in a) a.hasOwnProperty(u) && (s[u] = !0);
                e.exports = a
            },
            1: e => {
                "use strict";
                var t = function () {
                    function e(e, t) {
                        for (var n = 0; n < t.length; n++) {
                            var r = t[n];
                            r.enumerable = r.enumerable || !1, r.configurable = !0, "value" in r && (r.writable = !0), Object.defineProperty(e, r.key, r)
                        }
                    }
                    return function (t, n, r) {
                        return n && e(t.prototype, n), r && e(t, r), t
                    }
                }();

                function n(e) {
                    if (Array.isArray(e)) {
                        for (var t = 0, n = Array(e.length); t < e.length; t++) n[t] = e[t];
                        return n
                    }
                    return Array.from(e)
                }

                function r(e) {
                    return Array.isArray(e) ? e : Array.from(e)
                }
                var i = function () {
                    function e() {
                        ! function (e, t) {
                            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
                        }(this, e), this.replyId = 0, this.replies = {}, this.listeners = {}, this.hook = this.nullHook, this.debugLevel = 0, this.useTarget = !1, this.logger = console, this.posts = {}
                    }
                    return t(e, [{
                        key: "setPost",
                        value: function (e, t) {
                            "string" == typeof e ? this.posts[e] = t : this.post = e
                        }
                    }, {
                        key: "setUseTarget",
                        value: function (e) {
                            this.useTarget = e
                        }
                    }, {
                        key: "setDebugLevel",
                        value: function (e) {
                            this.debugLevel = e
                        }
                    }, {
                        key: "setHook",
                        value: function (e) {
                            var t = this,
                                n = Date.now();
                            this.hook = e ? function (r) {
                                r.timestamp = "undefined" != typeof window && void 0 !== window.performance ? window.performance.now() : Date.now() - n;
                                try {
                                    e(r)
                                } catch (e) {
                                    t.logger.warn("Hoor error", e)
                                }
                            } : this.nullHook
                        }
                    }, {
                        key: "nullHook",
                        value: function () {}
                    }, {
                        key: "call",
                        value: function () {
                            var e, t, i, o, a = this,
                                s = Array.prototype.slice.call(arguments);
                            if ("function" == typeof s[0] && (e = s.shift()), a.useTarget) {
                                var u = r(s);
                                t = u[0], i = u[1], o = u.slice(2)
                            } else {
                                var c = r(s);
                                i = c[0], o = c.slice(1)
                            }
                            return new Promise((function (r, s) {
                                var u = ++a.replyId;
                                a.debugLevel >= 2 && a.logger.info("rpc #" + u, "call =>", i, o), a.hook({
                                    type: "call",
                                    callee: t,
                                    rid: u,
                                    method: i,
                                    args: o
                                }), a.replies[u] = {
                                    resolve: r,
                                    reject: s,
                                    peer: t
                                };
                                var c = e || a.useTarget && a.posts[t] || a.post;
                                a.useTarget ? c(t, {
                                    type: "weh#rpc",
                                    _request: u,
                                    _method: i,
                                    _args: [].concat(n(o))
                                }) : c({
                                    type: "weh#rpc",
                                    _request: u,
                                    _method: i,
                                    _args: [].concat(n(o))
                                })
                            }))
                        }
                    }, {
                        key: "receive",
                        value: function (e, t, n) {
                            var r = this;
                            if (e._request) Promise.resolve().then((function () {
                                var t = r.listeners[e._method];
                                if ("function" == typeof t) return r.debugLevel >= 2 && r.logger.info("rpc #" + e._request, "serve <= ", e._method, e._args), r.hook({
                                    type: "call",
                                    caller: n,
                                    rid: e._request,
                                    method: e._method,
                                    args: e._args
                                }), Promise.resolve(t.apply(null, e._args)).then((function (t) {
                                    return r.hook({
                                        type: "reply",
                                        caller: n,
                                        rid: e._request,
                                        result: t
                                    }), t
                                })).catch((function (t) {
                                    throw r.hook({
                                        type: "reply",
                                        caller: n,
                                        rid: e._request,
                                        error: t.message
                                    }), t
                                }));
                                throw new Error("Method " + e._method + " is not a function")
                            })).then((function (n) {
                                r.debugLevel >= 2 && r.logger.info("rpc #" + e._request, "serve => ", n), t({
                                    type: "weh#rpc",
                                    _reply: e._request,
                                    _result: n
                                })
                            })).catch((function (n) {
                                r.debugLevel >= 1 && r.logger.info("rpc #" + e._request, "serve => !", n.message), t({
                                    type: "weh#rpc",
                                    _reply: e._request,
                                    _error: n.message
                                })
                            }));
                            else if (e._reply) {
                                var i = r.replies[e._reply];
                                delete r.replies[e._reply], i ? e._error ? (r.debugLevel >= 1 && r.logger.info("rpc #" + e._reply, "call <= !", e._error), r.hook({
                                    type: "reply",
                                    callee: i.peer,
                                    rid: e._reply,
                                    error: e._error
                                }), i.reject(new Error(e._error))) : (r.debugLevel >= 2 && r.logger.info("rpc #" + e._reply, "call <= ", e._result), r.hook({
                                    type: "reply",
                                    callee: i.peer,
                                    rid: e._reply,
                                    result: e._result
                                }), i.resolve(e._result)) : r.logger.error("Missing reply handler")
                            }
                        }
                    }, {
                        key: "listen",
                        value: function (e) {
                            Object.assign(this.listeners, e)
                        }
                    }]), e
                }();
                e.exports = new i
            },
            212: (e, t, n) => {
                "use strict";
                var r = function (e, t) {
                        if (Array.isArray(e)) return e;
                        if (Symbol.iterator in Object(e)) return function (e, t) {
                            var n = [],
                                r = !0,
                                i = !1,
                                o = void 0;
                            try {
                                for (var a, s = e[Symbol.iterator](); !(r = (a = s.next()).done) && (n.push(a.value), !t || n.length !== t); r = !0);
                            } catch (e) {
                                i = !0, o = e
                            } finally {
                                try {
                                    !r && s.return && s.return()
                                } finally {
                                    if (i) throw o
                                }
                            }
                            return n
                        }(e, t);
                        throw new TypeError("Invalid attempt to destructure non-iterable instance")
                    },
                    i = n(0),
                    o = n(1),
                    a = i.browser,
                    s = {},
                    u = {};

                function c(e, t) {
                    var n = !1;
                    return new Promise((function (t, r) {
                        return a.tabs.query({}).then((function (r) {
                            r.forEach((function (t) {
                                t.url === e && (a.tabs.update(t.id, {
                                    active: !0
                                }), a.windows.update(t.windowId, {
                                    focused: !0
                                }), n = !0)
                            })), t(n)
                        }))
                    }))
                }
                a.tabs.onRemoved.addListener((function (e) {
                    i.__closeByTab(e);
                    var t = u[e];
                    t && (delete u[e], delete s[t])
                })), e.exports = {
                    open: function (e, t) {
                        return "panel" === t.type ? function (e, t) {
                            return new Promise((function (n, o) {
                                c(a.runtime.getURL(t.url + "?panel=" + e)).then((function (n) {
                                    if (!n) return function (e, t) {
                                        return new Promise((function (n, o) {
                                            var c = a.runtime.getURL(t.url + "?panel=" + e);
                                            a.windows.getCurrent().then((function (l) {
                                                var d = t.width || 500,
                                                    f = t.height || 400,
                                                    p = {
                                                        url: c,
                                                        width: d,
                                                        height: f,
                                                        type: "popup",
                                                        left: Math.round((l.width - d) / 2 + l.left),
                                                        top: Math.round((l.height - f) / 2 + l.top)
                                                    };
                                                return i.isBrowser("chrome", "opera") && (p.focused = !0), a.windows.create(p).then((function (t) {
                                                    return s[e] = {
                                                        type: "window",
                                                        windowId: t.id
                                                    }, Promise.all([t, a.windows.update(t.id, {
                                                        focused: !0
                                                    })])
                                                })).then((function (s) {
                                                    var c = r(s, 1)[0];

                                                    function l(e) {
                                                        e != c.id && t.autoClose && a.windows.getCurrent().then((function (e) {
                                                            e.id != c.id && a.windows.remove(c.id).then((function () {}), (function () {}))
                                                        }))
                                                    }

                                                    function d(e) {
                                                        e == c.id && (a.windows.onFocusChanged.removeListener(l), a.windows.onFocusChanged.removeListener(d))
                                                    }
                                                    Promise.resolve().then((function () {
                                                        return t.initData && t.initData.autoResize ? void 0 : a.windows.update(c.id, {
                                                            height: c.height + 1
                                                        }).then((function () {
                                                            return a.windows.update(c.id, {
                                                                height: c.height - 1
                                                            })
                                                        }))
                                                    })).then((function () {
                                                        var e = new Promise((function (e, t) {
                                                                var n = setTimeout((function () {
                                                                    a.tabs.onCreated.removeListener(r), t(new Error("Tab did not open"))
                                                                }), 5e3);

                                                                function r(t) {
                                                                    t.windowId == c.id && (clearTimeout(n), a.tabs.onCreated.removeListener(r), e(t))
                                                                }
                                                                a.tabs.onCreated.addListener(r)
                                                            })),
                                                            t = a.tabs.query({
                                                                windowId: c.id
                                                            }).then((function (e) {
                                                                return new Promise((function (t, n) {
                                                                    e.length > 0 && t(e[0])
                                                                }))
                                                            }));
                                                        return Promise.race([e, t])
                                                    })).then((function (e) {
                                                        return "loading" == e.status ? new Promise((function (t, n) {
                                                            var r = setTimeout((function () {
                                                                a.tabs.onUpdated.removeListener(i), n(new Error("Tab did not complete"))
                                                            }), 6e4);

                                                            function i(n, o, s) {
                                                                n == e.id && "complete" == s.status && (clearTimeout(r), a.tabs.onUpdated.removeListener(i), t(s))
                                                            }
                                                            a.tabs.onUpdated.addListener(i)
                                                        })) : e
                                                    })).then((function (n) {
                                                        i.__declareAppTab(e, {
                                                            tab: n.id,
                                                            initData: t.initData
                                                        }), u[n.id] = e
                                                    })).then(n).catch(o), a.windows.onFocusChanged.addListener(l), a.windows.onRemoved.addListener(d)
                                                })).catch(o)
                                            })).catch(o)
                                        }))
                                    }(e, t)
                                })).then(n).catch(o)
                            }))
                        }(e, t) : function (e, t) {
                            return new Promise((function (n, r) {
                                var o = a.runtime.getURL(t.url + "?panel=" + e);
                                c(o).then((function (n) {
                                    if (!n) return a.tabs.create({
                                        url: o
                                    }).then((function (n) {
                                        i.__declareAppTab(e, {
                                            tab: n.id,
                                            initData: t.initData
                                        }), s[e] = {
                                            type: "tab",
                                            tabId: n.id
                                        }, u[n.id] = e
                                    }))
                                })).then(n).catch(r)
                            }))
                        }(e, t)
                    },
                    close: function (e) {
                        var t = s[e];
                        t && "tab" == t.type ? a.tabs.remove(t.tabId) : t && "window" == t.type ? a.windows.remove(t.windowId) : o.call(e, "close")
                    },
                    isOpen: function (e) {
                        return !!s[e]
                    }
                }
            },
            0: (e, t, n) => {
                "use strict";
                var r;
                t.browser = n(2), r = "undefined" == typeof browser && "undefined" != typeof chrome && chrome.runtime ? /\bOPR\//.test(navigator.userAgent) ? "opera" : "chrome" : /\bEdge\//.test(navigator.userAgent) ? "edge" : "firefox", t.browserType = r, void 0 === t.browser.action && (t.browser.action = t.browser.browserAction), t.isBrowser = function () {
                    for (var e = arguments.length, n = Array(e), r = 0; r < e; r++) n[r] = arguments[r];
                    for (var i = 0; i < n.length; i++)
                        if (n[i] == t.browserType) return !0;
                    return !1
                }, t.error = function (e) {
                    console.groupCollapsed(e.message), e.stack && console.error(e.stack), console.groupEnd()
                }
            }
        },
        t = {};

    function n(r) {
        var i = t[r];
        if (void 0 !== i) return i.exports;
        var o = t[r] = {
            id: r,
            loaded: !1,
            exports: {}
        };
        return e[r].call(o.exports, o, o.exports, n), o.loaded = !0, o.exports
    }
    n.d = (e, t) => {
        for (var r in t) n.o(t, r) && !n.o(e, r) && Object.defineProperty(e, r, {
            enumerable: !0,
            get: t[r]
        })
    }, n.g = function () {
        if ("object" == typeof globalThis) return globalThis;
        try {
            return this || new Function("return this")()
        } catch (e) {
            if ("object" == typeof window) return window
        }
    }(), n.hmd = e => ((e = Object.create(e)).children || (e.children = []), Object.defineProperty(e, "exports", {
        enumerable: !0,
        set: () => {
            throw new Error("ES Modules may not assign module.exports or exports.*, Use ESM export syntax, instead: " + e.id)
        }
    }), e), n.o = (e, t) => Object.prototype.hasOwnProperty.call(e, t), n.r = e => {
        "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(e, Symbol.toStringTag, {
            value: "Module"
        }), Object.defineProperty(e, "__esModule", {
            value: !0
        })
    }, (() => {
        "use strict";
        var e = n(3),
            t = e.browser,
            r = n(8),
            i = r.buildOptions || {};
        r.prod || console.info("=========== VDH started", (new Date).toLocaleTimeString(), "==========");
        var o = t.runtime.getManifest();
        e.prefs.declare(n(213)), n(214), n(34), n(191), n(226), n(225), n(197), JSON.parse(i.linuxlic || "false") && (e.prefs.linuxLicense = !0);
        var a = n(49);
        e.rpc.listen({
            openSettings: function () {
                e.ui.open("settings", {
                    type: "tab",
                    url: "content/settings.html"
                }), e.ui.close("main")
            },
            openTranslation: function () {
                e.ui.open("translation", {
                    type: "tab",
                    url: "content/translation.html"
                }), e.ui.close("main")
            },
            openSites: function () {
                return a.gotoOrOpenTab("https://www.downloadhelper.net/sites")
            },
            openForum: function () {
                return a.gotoOrOpenTab("https://groups.google.com/forum/#!forum/video-downloadhelper-q-and-a")
            },
            openHomepage: function () {
                return a.gotoOrOpenTab("https://www.downloadhelper.net/")
            },
            openTranslationForum: function () {
                return a.gotoOrOpenTab("https://groups.google.com/forum/#!forum/video-downloadhelper-internationalization")
            },
            openWeh: function () {
                return a.gotoOrOpenTab("https://github.com/mi-g/weh")
            },
            openAbout: function () {
                e.ui.open("about", {
                    type: "panel",
                    url: "content/about.html"
                }), e.ui.close("main")
            },
            openCoapp: function () {
                e.ui.open("coappShell", {
                    type: "tab",
                    url: "content/coapp-shell.html"
                }), e.ui.close("main")
            },
            goto: function (e) {
                return a.gotoOrOpenTab(e)
            },
            getBuild: function () {
                return r
            },
            updateLastFocusedWindowHeight: function (e, n) {
                t.windows.getLastFocused().then((function (r) {
                    if (r) {
                        var i = r.height - n;
                        t.windows.update(r.id, {
                            height: e + i
                        })
                    }
                }))
            }
        }), t.runtime.onInstalled.addListener((function (e) {
            "install" == e.reason ? a.gotoOrOpenTab("https://www.downloadhelper.net/welcome?browser=" + (i.browser || "") + "&version=" + o.version) : "update" != e.reason || e.previousVersion == t.runtime.getManifest().version || "7.2.1" == o.version || "7.2.2" == o.version || "7.3.1" == o.version && "7.3.0" == e.previousVersion || "7.3.3.0" == o.version && "7.3.3.1" == e.previousVersion || "7.3.3.1" == o.version && "7.3.3.2" == e.previousVersion || "7.4.0.1" == o.version && "7.4.0.0" == e.previousVersion || "7.5.0.0" == o.version || "8.0.0.6" == o.version || /^7\.3\.7(\.\d+)?$/.test(o.version) || a.gotoOrOpenTab("https://www.downloadhelper.net/update?browser=" + (i.browser || "") + "&from=" + e.previousVersion + "&to=" + o.version)
        }))
    })()
})();
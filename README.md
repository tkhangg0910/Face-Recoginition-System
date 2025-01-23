# Face Recognition System

## Installation

Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/tkhangg0910/Face-Recognition-System
    ```
2. Navigate to the project directory:
    ```bash
    cd src
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Setup Milvus in BE:
1. Navigate to the DB Backend directory:
    ```bash
    cd src/BE/db
    ```
2. Download the installation script and save it as `standalone.bat`:
    ```bash
    Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
    ```
3. Run the downloaded script to start Milvus as a Docker container:
    ```bash
    standalone.bat start
    ```
4. Run the container as needed.

---

## Architecture/Pipeline

Below is the pipeline architecture for the **Face Recognition System**:

```xml
<mxGraphModel>
    <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="2" value="" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#ED7100;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.container_registry_image;" vertex="1" parent="1">
            <mxGeometry x="-10" y="170" width="70" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="3" value="Vector Database" style="image;html=1;image=img/lib/clip_art/computers/Database_128x128.png" vertex="1" parent="1">
            <mxGeometry x="790" y="170" width="80" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="4" value="Face detector&lt;div&gt;(YOLOV11)&lt;/div&gt;" style="shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;" vertex="1" parent="1">
            <mxGeometry x="145" y="177.5" width="115" height="62.5" as="geometry"/>
        </mxCell>
        <mxCell id="5" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" source="6" target="8" parent="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="6" value="Landmark detector" style="shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;" vertex="1" parent="1">
            <mxGeometry x="150" y="285" width="120" height="65" as="geometry"/>
        </mxCell>
        <mxCell id="7" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" source="8" target="10" parent="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="8" value="Face Aligner" style="shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;" vertex="1" parent="1">
            <mxGeometry x="360" y="285" width="120" height="65" as="geometry"/>
        </mxCell>
        <mxCell id="9" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" source="10" target="13" parent="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="10" value="Face Embedder&lt;div&gt;(Inception Resnet)&lt;/div&gt;" style="shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;" vertex="1" parent="1">
            <mxGeometry x="352.5" y="177.5" width="135" height="65" as="geometry"/>
        </mxCell>
        <mxCell id="11" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" source="13" target="3" parent="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="12" value="Searching/Inserting" style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];" vertex="1" connectable="0" parent="11">
            <mxGeometry x="0.025" y="3" relative="1" as="geometry">
                <mxPoint as="offset"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="13" value="Embedding Vector" style="shape=image;html=1;verticalAlign=top;verticalLabelPosition=bottom;labelBackgroundColor=#ffffff;imageAspect=0;aspect=fixed;image=https://cdn1.iconfinder.com/data/icons/unicons-line-vol-6/24/vector-square-128.png" vertex="1" parent="1">
            <mxGeometry x="550" y="180" width="60" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="14" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=-0.008;entryY=0.431;entryDx=0;entryDy=0;entryPerimeter=0;" edge="1" source="2" target="4" parent="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="15" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.439;entryY=0.045;entryDx=0;entryDy=0;entryPerimeter=0;" edge="1" source="4" target="6" parent="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="16" value="Back End" style="rounded=1;whiteSpace=wrap;html=1;gradientColor=none;fillColor=none;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;" vertex="1" parent="1">
            <mxGeometry x="120" y="130" width="390" height="270" as="geometry"/>
        </mxCell>
    </root>
</mxGraphModel>
